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

    def prepare_data(self, data):
        batch_size = tf.shape(data['formula']).numpy()[0]
        return tf.zeros([batch_size, 0])

    def normalize(self, data_frame, approach=''):
        data_frame = data_frame.sample(
            min(data_frame.shape[0], self.kwargs['max_len_datasets'])
        )
        try:
            if approach == "abs_max_value":
                array = data_frame.to_numpy()
                max_value = np.abs(array.max())
                min_value = np.abs(array.min())
                abs_value = max(max_value, min_value)
                norm_array = np.divide(array, abs_value, dtype=np.float32)
            elif approach == "row_wise":
                measurements_64 = data_frame.to_numpy(dtype='float64', copy=True)
                norm_measurements_64 = sk_prepros.normalize(measurements_64, axis=1)
                norm_array = np.array(norm_measurements_64, dtype='float32')
            elif approach == "abs_max_y":
                array = data_frame.to_numpy(dtype='float32', copy=True)
                y = array[:, -1]
                max_value = np.abs(y.max())
                min_value = np.abs(y.min())
                abs_value = max(max_value, min_value)
                y_norm = y / abs_value
                array[:, -1] = y_norm
                norm_array = array
            elif approach == "clip":
                array = data_frame.to_numpy(dtype='float32', copy=True)
                norm_array = np.clip(array, a_min=-1, a_max=1)
            else:
                return data_frame
        except FloatingPointError:
            print(f'FloatingPointError happened in normalizing {array}')
            norm_array = np.zeros(shape=array.shape)

        if not np.all(np.isfinite(norm_array)):
            print('On of the Elements after the normalization is not an number. All of them are set to 0')
            norm_array = np.nan_to_num(norm_array, nan=np.float32(0.0), posinf=np.float32(3.4028235e+38), neginf=np.float32(-3.4028235e+38))
            print(f"After error handling the array is f{norm_array}")
        norm_dataframe = pd.DataFrame(norm_array, columns=data_frame.columns, dtype=np.float32)

        return norm_dataframe

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
