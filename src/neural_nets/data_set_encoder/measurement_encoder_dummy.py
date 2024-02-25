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

    def prepare_data(self, data):
        batch_size = tf.shape(data['data_frame']).numpy()
        output_size = [1] + list(batch_size)
        return tf.zeros(output_size)

    def normalize(self, data_frame):
        data_frame = data_frame.sample(
            min(data_frame.shape[0], self.kwargs['max_len_datasets'])
        )
        if not (self.norm_lin_transform or self.norm_abs_max_y):
            return data_frame
        try:
            output_columns = self.get_output_columns(data_frame)
            array = data_frame.to_numpy(dtype='float32', copy=True)
            if self.norm_abs_max_y:
                y = array[:, -1]
                max_value = np.abs(y.max())
                min_value = np.abs(y.min())
                abs_value = max(max_value, min_value, 1)
                y_norm = y / abs_value
                array[:, -1] = y_norm
            if self.norm_lin_transform:
                input_variables = array[:, :-1]
                x_max = input_variables.max()
                x_min = input_variables.min()
                data_range = max(x_max - x_min, 10e-6)
                a = 2 / data_range
                b = -2 * x_min / data_range - 1
                norm_input_variables = a * input_variables + b
                array[:, :-1] = norm_input_variables
                a_array = np.full((array.shape[0], 1), a)
                b_array = np.full((array.shape[0], 1), b)
                array = np.concatenate([a_array, b_array, array], axis=1)

        except FloatingPointError:
            print(f'FloatingPointError happened in normalizing {array}')
            output_shape = self.get_output_shape(data_frame)
            array = np.zeros(shape=output_shape)

        if not np.all(np.isfinite(array)):
            print(f"Before error handling the array is \n {array}")
            print('On of the Elements after the normalization is not an number. All of them are set to 0')
            array = np.nan_to_num(array, nan=np.float32(0.0), posinf=np.float32(3.4028235e+38), neginf=np.float32(-3.4028235e+38))
            print(f"After error handling the array is \n {array}")
        norm_dataframe = pd.DataFrame(array, columns=output_columns, dtype=np.float32)

        return norm_dataframe

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
