import tensorflow as tf


class MLP_Decoder(tf.Module):
    def __init__(self, *args, **kwargs):
        # Initialize the dimensions and activation functions
        self.out_dim = kwargs['out_dim']
        self.batch_sz = kwargs['batch_sz']
        self.normalize_way = kwargs['normalize_way']
        self.name_of_net = kwargs['name']
        #self.ln = tf.keras.layers.LayerNormalization(axis=1, name =f"{kwargs['name']}_layer_norm")

        self.dense_1 = tf.keras.layers.Dense(units=64, activation='relu', dtype=tf.float32, name =f"{kwargs['name']}_Dense_1")
        self.dense_2 = tf.keras.layers.Dense(units=64, activation='relu', dtype=tf.float32, name =f"{kwargs['name']}_Dense_2")
        self.dense_3 = tf.keras.layers.Dense(units=self.out_dim, dtype=tf.float32, name =f"{kwargs['name']}_Dense_3")
        self.softmax = tf.keras.layers.Softmax(
            axis=-1, name =f"{kwargs['name']}_Softmax"
        )

    def __call__(self, x, training):
        #x = self.ln(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        if self.normalize_way == 'soft_max':
            x = self.softmax(x)
        elif self.normalize_way == 'positive_sum_1':
            # Find the smallest value in each row
            min_values = tf.reduce_min(x, axis=1, keepdims=True)
            # Add the smallest value to each element in the corresponding row
            x = x - min_values
            # Calculate the sum of each row
            row_sums = tf.reduce_sum(x, axis=1, keepdims=True)
            # Divide each element in a row by the sum of that row
            x = x / (row_sums +  1e-08)
        elif self.normalize_way == 'sigmoid':
            x = tf.keras.activations.sigmoid(
                x
            )
        elif self.normalize_way == 'tanh':
            x = tf.math.tanh(x)
        elif self.normalize_way  == 'None':
            x=x
        else:
            raise AssertionError(f"{self.normalize_way} not defined as normalization")
        return x

    def __str__(self):
        return f"MLP_{self.name_of_net}"
