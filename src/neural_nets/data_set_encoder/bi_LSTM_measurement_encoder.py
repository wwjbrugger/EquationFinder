import tensorflow as tf
from src.neural_nets.data_set_encoder.LSTM_measurement_encoder import LstmEncoder

class BiLSTMEncoder(LstmEncoder):
    def __init__(self, *args, **kwargs):
        super(LstmEncoder, self).__init__(*args, **kwargs)
        self.encoder_measurements_LSTM_units = kwargs['encoder_measurements_LSTM_units']
        self.encoder_measurements_LSTM_return_sequence = kwargs['encoder_measurements_LSTM_return_sequence']

        self.bi_lstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=self.encoder_measurements_LSTM_units,
                                 return_sequences=self.encoder_measurements_LSTM_return_sequence,
                                 #return_state=True,
                                 #recurrent_initializer='glorot_uniform'
                                 )
        )

    def call(self, x, *args, **kwargs):
        #initial_state = self.initialize_hidden_state(batch_size=x.shape[0])
        output= self.bi_lstm_layer(
            x,
            #initial_state=initial_state,
            training=kwargs['training']
        )

        if self.encoder_measurements_LSTM_return_sequence == True:
            output = tf.math.reduce_max(output, axis=-2)
        return output
