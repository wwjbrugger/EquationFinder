import tensorflow as tf
from src.utils.tensors import check_for_non_numeric_and_replace_by_0
from src.utils.logging import get_log_obj
from src.contrastive_loss.contrastive_loss import (
    prepare_for_contrastive_loss, postprocess_contrastive_loss)
class RulePredictorNet(tf.keras.Model):

    def __init__(self, encoder_tree_class, encoder_tree_args,
                 encoder_measurement_class, encoder_measurement_args,
                 actor_decoder_class, actor_decoder_args,
                 critic_decoder_class, critic_decoder_args,
                 args):
        super(RulePredictorNet, self).__init__()
        self.encoder_tree = encoder_tree_class(**encoder_tree_args)
        self.encoder_measurement = encoder_measurement_class(**encoder_measurement_args)
        self.actor = actor_decoder_class(**actor_decoder_args)
        self.critic = critic_decoder_class(**critic_decoder_args)

        self.training = None
        self.args = args
        self.logger_net = get_log_obj(args=args, name='RulePredictorNet')
        if self.args.contrastive_loss:
            # For contrastive loss the rows of the dataset has to be split in two parts.
            # when using the DatasetTransformer the shape of a sample is [batch, column, row, encoding ]
            # in all other cases it is [batch, row,  column, encoding ]
            self.axis_to_split = 2 if self.args.class_measurement_encoder == 'DatasetTransformer' else 1

    @tf.function
    def __call__(self, input_encoder_tree, input_encoder_measurement):
        input_encoder_tree = check_for_non_numeric_and_replace_by_0(
            logger=self.logger_net,
            tensor=input_encoder_tree,
            name='tree_representation'
        )
        input_encoder_measurement = check_for_non_numeric_and_replace_by_0(
            logger=self.logger_net,
            tensor=input_encoder_measurement,
            name='measurement_representation'
        )
        encoding_tree = self.encoder_tree(
            x=input_encoder_tree,
            training=self.training
        )

        if self.args.contrastive_loss:
            input_encoder_contrastive = prepare_for_contrastive_loss(
                input_encoder_measurement=input_encoder_measurement,
                axis_to_split=  self.axis_to_split
            )
            encoding_measurement_contrastive = self.encoder_measurement(
                x=input_encoder_contrastive,
                training=self.training
            )
            encoding_measurement = postprocess_contrastive_loss(
                output_encoder_measurement=encoding_measurement_contrastive
            )
        else:
            encoding_measurement = self.encoder_measurement(
                x=input_encoder_measurement,
                training=self.training
            )
            encoding_measurement_contrastive = None

        output_encoder = tf.concat([encoding_tree, encoding_measurement], axis=1)
        # if self.i % 50 == 0:
        #     self.logger_net.info(f"Sum Output encoding_measurement: {np.sum(np.abs(encoding_measurement.numpy()))}")
        #     self.logger_net.info(f"Sum Output Norm encoding_measurement: {np.sum(np.abs(encoding_measurement_norm.numpy()))}")
        #
        #     self.logger_net.info(f"Encoding_tree sum: {np.sum(np.abs(encoding_tree.numpy()))}")
        #     self.logger_net.info(f"Encoding_tree Norm sum: {np.sum(np.abs(encoding_tree_norm.numpy()))}")

        action_uncliped = self.actor(
            x=output_encoder,
            training=self.training
        )
        critic_uncliped = self.critic(
            x=output_encoder,
            training=self.training
        )
        return (action_uncliped, critic_uncliped,
                encoding_measurement_contrastive, input_encoder_contrastive)


