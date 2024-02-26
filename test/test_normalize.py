import numpy as np

from src.neural_nets.data_set_encoder.measurement_encoder_dummy import MeasurementEncoderDummy
import unittest
import random
import pandas as pd


class TestNormalization(unittest.TestCase):
    def setUp(self) -> None:
        d = {'col1': [0, 1, 2, 3], 'col2': [0, 1, 4, 9]}
        self.data_frame = pd.DataFrame(data=d, index=[0, 1, 2, 3])
        np.random.seed(0)

    def test_normalize_abs_max_y(self):
        kwargs = {
            'max_len_datasets': 250,
            'normalize_approach': 'abs_max_y',
        }
        self.measurement_encoder = MeasurementEncoderDummy(**kwargs)
        norm_dataframe = self.measurement_encoder.normalize(data_frame=self.data_frame)
        np.testing.assert_array_almost_equal(norm_dataframe.to_numpy(),
                                             np.array(
                                                 [[2, 0.44444445],
                                                  [3, 1.],
                                                  [1, 0.11111111],
                                                  [0., 0.]],
                                                 dtype=np.float32
                                             )
                                             )

    def test_infinity(self):
        d = {'col1': [1, 2], 'col2': [3.4028235e+70, -3.4028235e+38]}
        self.data_frame = pd.DataFrame(data=d, index=[0, 1], dtype=np.float32)
        kwargs = {
            'max_len_datasets': 250,
            'normalize_approach': 'abs_max_y',
        }
        self.measurement_encoder = MeasurementEncoderDummy(**kwargs)

        norm_dataframe = self.measurement_encoder.normalize(data_frame=self.data_frame)
        np.testing.assert_array_almost_equal(norm_dataframe.to_numpy(),
                                             np.array([[2., 0.],
                                                       [1, 0]],
                                                      dtype=np.float32
                                                      )
                                             )

    def test_underflow(self):
        np.seterr(all='raise')
        d = {'col1': [1, np.pi], 'col2': [1e-40, -10]}
        self.data_frame = pd.DataFrame(data=d, index=[0, 1], dtype=np.float32)
        kwargs = {
            'max_len_datasets': 250,
            'normalize_approach': 'abs_max_y',
        }
        self.measurement_encoder = MeasurementEncoderDummy(**kwargs)

        norm_dataframe = self.measurement_encoder.normalize(data_frame=self.data_frame)
        pass
        np.testing.assert_array_almost_equal(norm_dataframe.to_numpy(),
                                             np.array([[0., 0.],
                                                       [0, 0]],
                                                      dtype=np.float32
                                                      )
                                             )

    def test_lin_transformation(self):
        np.seterr(all='raise')
        kwargs = {
            'max_len_datasets': 250,
            'normalize_approach': 'abs_max_y__lin_transform',
        }
        self.measurement_encoder = MeasurementEncoderDummy(**kwargs)

        norm_dataframe = self.measurement_encoder.normalize(data_frame=self.data_frame)
        pass
        np.testing.assert_array_almost_equal(norm_dataframe.to_numpy(),
                                             np.array(
                                                 [[0.667, -1., 0.333, 0.444],
                                                  [0.667, - 1., 1., 1.],
                                                  [0.667, - 1., - 0.333, 0.111],
                                                  [0.667, - 1., - 1., 0.]],
                                                 dtype=np.float32
                                             ),
                                             decimal = 2)
