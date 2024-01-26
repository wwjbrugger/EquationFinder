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

    def prepare_data(self, data):
        norm_frame = self.normalize(
            data_frame=data['data_frame'],
            approach=self.kwargs['normalize_approach']
        )
        tensor_one_row = tf.reshape(tf.convert_to_tensor(norm_frame, dtype=tf.float32), -1)
        tensor_one_row = tf.expand_dims(tensor_one_row, axis=0)
        return tensor_one_row

    def call(self, x, *args, **kwargs):
        output = x
        for layer in self.class_layers:
            output = layer(output, training= kwargs['training'])
        return output
