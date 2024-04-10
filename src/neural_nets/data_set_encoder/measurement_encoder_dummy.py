import numpy as np
import tensorflow as tf
import pandas as pd
import sklearn.preprocessing as sk_prepros
from src.utils.error import NonFiniteError



class MeasurementEncoderDummy(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(MeasurementEncoderDummy, self).__init__()
        self.output_dim = 0
        self.column_names = []
        self.kwargs = kwargs
        self.norm_abs_max_y ="abs_max_y" in kwargs['normalize_approach']
        self.norm_lin_transform = "lin_transform" in kwargs['normalize_approach']

    def get_output_shape(self, data_frame):
        input_shape = data_frame.shape
        if self.norm_lin_transform:
            return (input_shape[0], input_shape[1] +2 )
        else:
            return (input_shape[0], input_shape[1])

    def get_output_columns(self, data_frame):
        input_columns = list(data_frame.columns)
        if self.norm_lin_transform:
            return ['a','b'] + input_columns
        else:
            return input_columns

    def reshape_measurements(self, data):
        tensor_with_row = reshape_measurements(
            data=data,
            column_names=self.column_names,
            measurement_in_row=self.measurement_in_row
        )
        return tensor_with_row

    def call(self, x, *args, **kwargs):
        shape = tf.shape(x)
        empty_tensor = tf.zeros([shape[0], 0])
        return empty_tensor


def reshape_measurements(data, column_names, measurement_in_row):
    stacked_tensor = tf.stack([data[feature] for feature in column_names], axis=1)
    batch_size = tf.shape(stacked_tensor)[0]
    stacked_tensor = tf.squeeze(stacked_tensor)
    tensor_with_row = tf.reshape(tensor=stacked_tensor,
                                 shape=(batch_size, -1, measurement_in_row)
                                 )
    return tensor_with_row


def reshape_measurements_to_pandas(data, column_names, measurement_in_row):
    columns = [feature.split('_row_')[0] for feature in column_names if 'row_0' in feature]
    data_set = np.array([data[feature].numpy()[0] for feature in column_names])
    data_set = np.reshape(data_set, (-1, len(columns)))
    data_set = pd.DataFrame(data=data_set, columns=columns)
    return data_set
