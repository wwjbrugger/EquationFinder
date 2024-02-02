import tensorflow as tf
from src.neural_nets.data_set_encoder.measurement_encoder_dummy import MeasurementEncoderDummy


class LstmEncoder(MeasurementEncoderDummy):
    def __init__(self, *args, **kwargs):
        super(LstmEncoder, self).__init__(*args, **kwargs)
        self.encoder_measurements_LSTM_units = kwargs['encoder_measurements_LSTM_units']
        self.encoder_measurements_LSTM_return_sequence = kwargs['encoder_measurements_LSTM_return_sequence']

        self.lstm_layer = tf.keras.layers.LSTM(units=self.encoder_measurements_LSTM_units,
                                               return_sequences=self.encoder_measurements_LSTM_return_sequence,
                                               return_state=True,
                                               recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self, batch_size):
        return [tf.zeros((batch_size, self.encoder_measurements_LSTM_units)),
                tf.zeros((batch_size, self.encoder_measurements_LSTM_units))]

    def prepare_data(self, data):
        norm_frame = self.normalize(data_frame=data['data_frame'])

        tensor_with_row= tf.expand_dims(tf.convert_to_tensor(norm_frame, dtype=tf.float32), axis=0)
        return tensor_with_row

    def call(self, x, *args, **kwargs):
        x = tf.cast(x, dtype=tf.float32)
        output, h, c = self.lstm_layer(
            x,
            initial_state=self.initialize_hidden_state(batch_size=x.shape[0]),
            training=kwargs['training']
        )
        if self.encoder_measurements_LSTM_return_sequence == True:
            output = tf.math.reduce_max(output, axis=-2)
        return output
