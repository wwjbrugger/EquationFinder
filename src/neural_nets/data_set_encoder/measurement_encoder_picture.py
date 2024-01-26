import tensorflow as tf
import numpy as np
from src.neural_nets.data_set_encoder.measurement_encoder_dummy import MeasurementEncoderDummy


# based on  https://github.com/OATML/non-parametric-transformers Kossen et al., “Self-Attention Between Datapoints.” and
# https://github.com/arrigonialberto86/set_transformer a tensorflow implementation of Zaheer et al., “Deep Sets.”
class MeasurementEncoderPicture(MeasurementEncoderDummy):
    def __init__(self,*args, **kwargs):
        super(MeasurementEncoderDummy, self).__init__()
        self.kwargs = kwargs

        self.conv_0= tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                   padding='same', strides=2,
                                   data_format='channels_first',
                                            name="Convolution_0_Encoder_Picture")
        self.conv_1= tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                   padding='same', strides=2,
                                   data_format='channels_first',
                                            name="Convolution_1_Encoder_Picture"
                                            )
        self.conv_2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                             padding='same', strides=1,
                                             data_format='channels_first',
                                             name="Convolution_2_Encoder_Picture"
                                             )
        self.conv_3 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu',
                                             padding='same', strides=1,
                                             data_format='channels_first',
                                             name="Convolution_3_Encoder_Picture"
                                             )
        self.last_layer_0 = tf.keras.layers.Dense(units=128, activation='gelu',
                                                  name='Dense_0_Encoder_Picture'
                                                  )
        self.last_layer_1 = tf.keras.layers.Dense(units=64, activation='gelu',
                                                  name='Dense_1_Encoder_Picture'
                                                  )
        self.last_layer_2 = tf.keras.layers.Dense(units=32,
                                                  name='Dense_2_Encoder_Picture'
                                                  )




    def __call__(self, x, *args, **kwargs):
        x_picture = table_to_picture(x, bins=14, histogram_range=[[-10., 10.], [-1., 1.]])
        after_conv_0 = self.conv_0(x_picture)
        after_conv_1 = self.conv_1(after_conv_0)
        after_conv_2 = self.conv_2(after_conv_1)
        after_conv_3 = self.conv_3(after_conv_2)
        X_flat = tf.reshape(after_conv_3, (after_conv_3.shape[0], -1))
        X_flat_0 = self.last_layer_0(X_flat)
        X_flat_1 = self.last_layer_1(X_flat_0)
        X_flat_2 = self.last_layer_2(X_flat_1)
        norm = tf.linalg.norm(X_flat_2, ord='euclidean', name=None, keepdims=True, axis=-1)
        out = X_flat_2 / norm
        return out



    def prepare_data(self, data):

        norm_frame = self.normalize(
            data_frame=data['data_frame'],
            approach=self.kwargs['normalize_approach']
        )
        tensor = tf.convert_to_tensor(norm_frame, dtype=tf.float32)


        # Dataset transformer expect each cell in table to be encoded.
        # we are not doing so we add an extra dimension at the end
        # add batch dimension
        tensor = tf.expand_dims(tensor, axis=0)
        return tensor

def table_to_picture(tensor, bins, histogram_range):
    batch_list = []
    for b in range(tensor.shape[0]):
        picture_list = []
        for c in range(tensor.shape[-1] -1):
            histogram = histogram2d(x=tensor[b,:,c],
                                       y=tensor[b,:,-1],
                                       nbins=bins,
                                       value_range=histogram_range,
                                       weights=None
                                       )
            picture_list.append(tf.where(histogram >=1, np.float32(1), np.float32(0)))
        batch_list.append(picture_list)
    picture_tensor = tf.convert_to_tensor(batch_list, dtype=tf.float32)
    return  picture_tensor


def histogram2d(x, y,
              value_range,
              nbins=100,
              weights=None,
              bin_dtype=tf.dtypes.float32):
    """
    Bins x, y coordinates of points onto simple square 2d histogram

    Given the tensor x and y:
    x: x coordinates of points
    y: y coordinates of points
    this operation returns a rank 2 `Tensor`
    representing the indices of a histogram into which each element
    of `values` would be binned. The bins are equal width and
    determined by the arguments `value_range` and `nbins`.


  Args:
    x: Numeric `Tensor`.
    y: Numeric `Tensor`.
    value_range[0] lims for x
    value_range[1] lims for y

    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
    weights: The value to scale
    dtype:  dtype for returned histogram.

  Example:
  N = 1000
  xs =  tf.random.normal([N])
  ys =  tf.random.normal([N])
  get2dHistogram(xs, ys, ([-5.0, 5.0], [-5.0, 5.0]),  50)


    """
    x_range = value_range[0]
    y_range = value_range[1]

    if weights is None:
        hist_bins = tf.histogram_fixed_width_bins(y, y_range, nbins=nbins, dtype=bin_dtype)
        return tf.map_fn(lambda i: tf.histogram_fixed_width(x[hist_bins == i], x_range, nbins=nbins),
                         tf.range(nbins))

    x_bins = tf.histogram_fixed_width_bins(x, x_range, nbins=nbins, dtype=bin_dtype)
    y_bins = tf.histogram_fixed_width_bins(y, y_range, nbins=nbins, dtype=bin_dtype)
    hist = tf.zeros((nbins, nbins), dtype=weights.dtype)
    indices = tf.transpose(tf.stack([y_bins, x_bins]))
    return tf.tensor_scatter_nd_add(hist, indices, weights)
