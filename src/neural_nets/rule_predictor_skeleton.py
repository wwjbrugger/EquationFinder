import pandas as pd
import tensorflow as tf
import typing
from src.neural_nets.get_rule_predictor_class import get_rule_predictor
import numpy as np
import src.neural_nets.loss as loss
from src.utils.tensors import expand_tensor_to_same_size
from src.utils.logging import get_log_obj
from src.utils.tensors import check_for_non_numeric_and_replace_by_0
from src.utils.tensors import tf_save_cast_to_float_32
import tensorboard
import time
from src.neural_nets.loss import NT_Xent



class RulePredictorSkeleton(tf.keras.Model):

    def __init__(self, args, reader_train):
        super(RulePredictorSkeleton, self).__init__()
        self.single_player = True,
        self.net = get_rule_predictor(
            args=args,
            reader_data=reader_train
        )
        self.optimizer_encoder_measurement = tf.keras.optimizers.Adam(
            learning_rate=1e-5,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9,
            clipnorm=0.001
        )
        self.optimizer_encoder_equation = tf.keras.optimizers.Adam(
        )
        self.optimizer_actor = tf.keras.optimizers.Adam(
        )
        self.optimizer_critic = tf.keras.optimizers.Adam(
        )
        self.args = args

        self.training = None
        self.steps = 0
        self.logger = get_log_obj(args=args, name='AlphaZeroRulePredictor')
        if self.args.contrastive_loss:
            self.contrastive_loss = NT_Xent(args=self.args)

    def train(self, examples: typing.List):
        """
                This function trains the neural network with data gathered from self-play.

                :param examples: a list of training examples of the form: (o_t, (pi_t, v_t), w_t)
                """
        self.net.training = True
        measurement_representation, target_pis, target_vs, tree_representation = \
            self.prepare_batch_for_NN(examples)
        target_dataset_encoding = self.get_target_dataset_encoding(examples)
        pi_batch_loss, v_batch_loss, contrastive_loss, encoding_measurement = self.train_step(
            measurement_representation=measurement_representation,
            target_pis=target_pis,
            target_vs=target_vs,
            tree_representation=tree_representation,
            target_dataset_encoding=target_dataset_encoding
        )

        self.steps += 1
        return pi_batch_loss, v_batch_loss, contrastive_loss

    def get_target_dataset_encoding(self, examples):
        column_true_equation = np.array([example['observation']['true_equation_hash'] for example in examples])
        row_true_equation = np.expand_dims(column_true_equation, axis=1)
        equality_matrix = np.equal(column_true_equation, row_true_equation)
        equality_matrix_contrastive = np.repeat(np.repeat(equality_matrix, axis=0, repeats=2), axis=1, repeats=2)
        target_dataset_encoding = pd.DataFrame(
            equality_matrix_contrastive,
            index=np.repeat(column_true_equation, 2),
            columns=np.repeat(column_true_equation, 2)
        )
        return target_dataset_encoding

    def prepare_batch_for_NN(self, examples):
        observations, loss_scale, target_pis, target_vs = [], [], [], []
        for example in examples:
            observations.append(example['observation'])
            if 'probabilities_actor' in example:
                target_pis.append(example['probabilities_actor'])
                target_vs.append(example['observed_return'])
                loss_scale.append(example['loss_scale'])
        tree_representation_list = [
            self.get_tree_representation(observation)[0] for observation in
            observations]

        measurement_representation_list = [
            observation['data_frame'] for observation in observations
        ]
        target_pis = tf_save_cast_to_float_32(
            x=target_pis,
            logger=self.logger,
            name='target_pis'
        )
        target_vs = tf_save_cast_to_float_32(
            x=target_vs,
            logger=self.logger,
            name='target_vs'
        )

        return measurement_representation_list, target_pis, target_vs, tree_representation_list

    @tf.function
    def train_step(self, measurement_representation, target_pis,
                   target_vs, tree_representation, target_dataset_encoding=None):
        with tf.GradientTape(persistent=True) as tape:
            action_prediction, v, encoding_measurement, split_measurement = self.net(
                input_encoder_tree=tree_representation,
                input_encoder_measurement=measurement_representation
            )

            if self.args.contrastive_loss:
                loss_measurement_encoder = self.contrastive_loss(
                    zizj=encoding_measurement,
                    target_dataset_encoding=target_dataset_encoding
                )
            else:
                loss_measurement_encoder = 0

            pi_batch_loss = loss.kl_divergence(
                real=target_pis,
                pred=action_prediction
            )
            v_batch_loss = loss.mean_square_error_loss_function(
                real=target_vs,
                pred=v
            )
        if self.args.path_to_pretrained_dataset_encoder is None:
            variables_encoder_measurements = [resourceVariable for resourceVariable in
                                              self.net.encoder_measurement.trainable_variables]
            if self.args.contrastive_loss:
                gradients_encoder_measurements = tape.gradient(
                    loss_measurement_encoder,
                    variables_encoder_measurements
                )
            else:
                gradients_encoder_measurements = tape.gradient(
                    pi_batch_loss,
                    variables_encoder_measurements
                )
            gradients_encoder_measurements = [check_for_non_numeric_and_replace_by_0(
                logger=self.logger, tensor=x, name='target_pis'
            ) for x in gradients_encoder_measurements]
            self.optimizer_encoder_measurement.apply_gradients(
                zip(gradients_encoder_measurements, variables_encoder_measurements)
            )

        variables_encoder_tree = [resourceVariable for resourceVariable in
                                  self.net.encoder_tree.trainable_variables]
        gradients_encoder_tree = tape.gradient(pi_batch_loss, variables_encoder_tree)
        gradients_encoder_tree = [check_for_non_numeric_and_replace_by_0(
            logger=self.logger, tensor=x, name='target_pis'
        ) for x in gradients_encoder_tree]
        self.optimizer_encoder_equation.apply_gradients(zip(gradients_encoder_tree, variables_encoder_tree))

        variables_actor = [resourceVariable for resourceVariable in
                           self.net.actor.trainable_variables]
        gradients_actor = tape.gradient(pi_batch_loss, variables_actor)
        gradients_actor = [check_for_non_numeric_and_replace_by_0(
            logger=self.logger, tensor=x, name='target_pis'
        ) for x in gradients_actor]
        self.optimizer_actor.apply_gradients(zip(gradients_actor, variables_actor))

        variables_critic = [resourceVariable for resourceVariable in
                            self.net.critic.trainable_variables]
        gradients_critic = tape.gradient(v_batch_loss, variables_critic)
        gradients_critic = [check_for_non_numeric_and_replace_by_0(
            logger=self.logger, tensor=x, name='target_pis'
        ) for x in gradients_critic]
        self.optimizer_critic.apply_gradients(zip(gradients_critic, variables_critic))

        return pi_batch_loss, v_batch_loss, loss_measurement_encoder, encoding_measurement

    def predict(self, examples):
        action_prediction, v, _, _, _ = self.predict_with_loss(examples, with_loss=False)
        return action_prediction, v

    def predict_with_loss(self, examples, with_loss=True):
        self.net.training = False
        measurement_representation, target_pis, target_vs, tree_representation = \
            self.prepare_batch_for_NN(examples)
        target_dataset_encoding = self.get_target_dataset_encoding(examples)
        action_prediction, v, encoding_measurement, split_measurement = self.net(
            input_encoder_tree=tree_representation,
            input_encoder_measurement=measurement_representation
        )
        if with_loss:
            pi_batch_loss = loss.kl_divergence(
                real=target_pis,
                pred=action_prediction
            )
            v_batch_loss = loss.mean_square_error_loss_function(
                real=target_vs,
                pred=v
            )
            if self.args.contrastive_loss:
                loss_measurement_encoder = self.contrastive_loss(
                    zizj=encoding_measurement,
                    target_dataset_encoding=target_dataset_encoding
                )
            else:
                loss_measurement_encoder = 0
        else:
            pi_batch_loss = None
            v_batch_loss = None
            loss_measurement_encoder = None

        action_prediction = tf.squeeze(action_prediction).numpy().astype(np.float32)
        v = tf.squeeze(v).numpy().astype(np.float32)
        return action_prediction, v, pi_batch_loss, v_batch_loss, loss_measurement_encoder

    def get_tree_representation(self, network_input):
        tree_representation = \
            network_input['current_tree_representation_int']
        if self.args.use_position_encoding:
            node_to_expand = tf.cast(network_input['id_last_node'], np.float32)
            node_to_expand = expand_tensor_to_same_size(
                to_change=node_to_expand,
                reference=tree_representation
            )

            tree_representation = tf.concat(
                [tree_representation, node_to_expand],
                axis=1)
        return tree_representation
