import tensorflow as tf
from src.utils.tensors import check_for_non_numeric_and_replace_by_0
from src.utils.logging import get_log_obj
from src.contrastive_loss.contrastive_loss import (
    prepare_for_contrastive_loss, postprocess_contrastive_loss)
from src.utils.tensors import tf_save_cast_to_float_32


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
        self.norm_abs_max_y = "abs_max_y" in args.normalize_approach
        self.norm_lin_transform = "lin_transform" in args.normalize_approach
        if self.args.contrastive_loss:
            # For contrastive loss the rows of the dataset has to be split in two parts.
            # when using the DatasetTransformer the shape of a sample is [batch, column, row, encoding ]
            # in all other cases it is [batch, row,  column, encoding ]
            self.axis_to_split = 2 if self.args.class_measurement_encoder == 'DatasetTransformer' else 1

    @tf.function
    def __call__(self, input_encoder_tree, input_encoder_measurement):
        """

        :param input_encoder_tree: list of np_lists
        :param input_encoder_measurement: list of pandas frames
        :return:
        """
        input_encoder_measurement_old = input_encoder_measurement
        input_encoder_measurement = tf.convert_to_tensor(
            [tf_save_cast_to_float_32(frame,
                                      self.logger_net,
                                      'convert measurements')
             for frame in input_encoder_measurement
             ]
        )

        input_encoder_tree = tf.convert_to_tensor(
            [tf_save_cast_to_float_32(array,
                                      self.logger_net,
                                      'convert syntax tree')
             for array in input_encoder_tree]
        )

        if self.args.contrastive_loss:
            input_encoder_measurement = prepare_for_contrastive_loss(
                input_encoder_measurement=input_encoder_measurement,
                axis_to_split=self.axis_to_split
            )

        input_encoder_measurement = tf.map_fn(self.scale_measurements,
                                              input_encoder_measurement)
        encoding_measurement = self.encoder_measurement(
            x=input_encoder_measurement,
            training=self.training
        )
        if self.args.contrastive_loss:
            encoding_measurement_contrastive = encoding_measurement
            encoding_measurement = postprocess_contrastive_loss(
                output_encoder_measurement=encoding_measurement
            )
        else:
            encoding_measurement_contrastive = None

        encoding_tree = self.encoder_tree(
            x=input_encoder_tree,
            training=self.training
        )
        output_encoder = tf.concat([encoding_tree, encoding_measurement], axis=1)

        action_uncliped = self.actor(
            x=output_encoder,
            training=self.training
        )
        critic_uncliped = self.critic(
            x=output_encoder,
            training=self.training
        )
        return (action_uncliped, critic_uncliped,
                encoding_measurement_contrastive,
                input_encoder_measurement)

    def scale_measurements(self, tensor):
        max_elements = tf.minimum(
            tf.shape(tensor)[0],
            self.args.max_len_datasets
        )
        index = tf.random.shuffle(tf.range(tf.shape(tensor)[0]))[:max_elements]
        tensor_random = tf.gather(tensor, indices=index, axis=0)
        if not (self.norm_lin_transform or self.norm_abs_max_y):
            return tensor_random
        try:
            output_shape = (tf.shape(tensor_random)[0],
                            tf.shape(tensor_random)[1] + 2
                            if self.norm_lin_transform
                            else tf.shape(tensor_random)[1]
                            )
            y = tensor_random[:, -1]
            input_variables = tensor_random[:, :-1]
            if self.norm_abs_max_y:
                max_value = tf.math.abs(tf.math.reduce_max(y))
                min_value = tf.math.abs(tf.math.reduce_min(y))
                abs_value = tf.math.reduce_max([max_value, min_value, 1])
                y = tf.math.divide(y, abs_value)

            if self.norm_lin_transform:
                x_max = tf.math.reduce_max(input_variables)
                x_min = tf.math.reduce_min(input_variables)
                data_range = tf.math.reduce_max([x_max - x_min, 10e-6])
                a = tf.math.divide(2, data_range)
                b = tf.math.divide(-2 * x_min, data_range) - 1
                norm_input_variables = tf.math.multiply(a, input_variables) + b
                a_array = tf.fill([tf.shape(norm_input_variables)[0], 1], a)
                b_array = tf.fill([tf.shape(norm_input_variables)[0], 1], b)
                input_variables = tf.concat(
                    [a_array, b_array, norm_input_variables], axis=1
                )

            tensor_random = tf.concat(
                [input_variables, tf.expand_dims(y, 1)],
                axis=1
            )

        except FloatingPointError:
            print(f'FloatingPointError happened in normalizing {tensor}')
            tensor_random = tf.fill(
                output_shape,
                tf.convert_to_tensor(0, dtype=tf.float32)
            )

        tensor_random = check_for_non_numeric_and_replace_by_0(
            tensor=tensor_random,
            logger=self.logger_net,
            name='scale_measurements'
        )
        arg_sort_y = tf.argsort(tensor_random[:, -1])
        tensor_random = tf.gather(tensor_random, indices=arg_sort_y, axis=0)

        return tensor_random
