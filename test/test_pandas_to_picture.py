import numpy as np

from src.neural_nets.data_set_encoder.measurement_encoder_picture import table_to_picture, MeasurementEncoderPicture
import unittest
import random
import pandas as pd
import tensorflow as tf


class TestPandasToPicture(unittest.TestCase):
    def setUp(self) -> None:
        kwargs = {
            'max_len_datasets': 250
        }
        self.measurement_encoder = MeasurementEncoderPicture(**kwargs)
        d = {'x_0': [-10, -4, 4, 10], 'x_1': [-2, 1, 2, 2], 'y': [-1, -1, 1, 1]}
        self.data_frame = pd.DataFrame(data=d, index=[0, 1, 2, 3])
        np.random.seed(0)

    def test_pandas_to_pictures(self):
        tensor_df = tf.convert_to_tensor(self.data_frame, dtype=tf.float32)
        tensor_df = tf.expand_dims(tensor_df, axis=0)
        tensor = table_to_picture(tensor_df, bins=4)
        true_value = np.array([[[[1., 1., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 1., 1.]],
        [[0., 1., 1., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 1., 0.]]]], dtype='float32')
        np.testing.assert_array_equal(x=tensor.numpy(), y=true_value)
