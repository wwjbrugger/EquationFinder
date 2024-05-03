import tensorflow as tf
from src.neural_nets.data_set_encoder.measurement_encoder_dummy import MeasurementEncoderDummy


class  MlpMeasurementEncoder(MeasurementEncoderDummy):
    def __init__(self, *args, **kwargs):
        super(MlpMeasurementEncoder, self).__init__(*args, **kwargs)
        self.encoder_measurement_num_layer = kwargs['encoder_measurement_num_layer']
        self.encoder_measurement_num_neurons = kwargs['encoder_measurement_num_neurons']
        self.dropout_rate = kwargs['dropout_rate']


        self.class_layers = []
        for i in range(self.encoder_measurement_num_layer):
            self.class_layers.append(
                tf.keras.layers.Dense(
                    units=self.encoder_measurement_num_neurons,
                    activation='relu'
                )
            )
            self.class_layers.append(
                tf.keras.layers.Dropout(self.dropout_rate)
            )

    def call(self, x, *args, **kwargs):
        x_old = x
        x = tf.reshape(x, shape=[tf.shape(x)[0], -1])
        output = x
        for layer in self.class_layers:
            output = layer(output, training= kwargs['training'])
        norm = tf.linalg.norm(output, ord='euclidean', name=None, keepdims=True, axis=-1)
        out_norm = output / norm
        return out_norm

