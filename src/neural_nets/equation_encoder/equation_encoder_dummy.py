import tensorflow as tf


class EquationEncoderDummy(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(EquationEncoderDummy, self).__init__()
        self.output_dim = 0
        self.column_names = []

    def prepare_data(self, data):
        batch_size = tf.shape(data['infix_formula']).numpy()[0]
        return tf.zeros([batch_size, 0])

    def call(self, x, *args, **kwargs):
        shape = tf.shape(x)
        empty_tensor = tf.zeros([shape[0], 0])
        return empty_tensor




