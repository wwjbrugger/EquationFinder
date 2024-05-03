import tensorflow as tf
import numpy as np
def prepare_for_contrastive_loss( input_encoder_measurement, axis_to_split):
    """
    Splits a dataset into two pieces. So a tensor of Size [Batch, Rows, Columns, ValueEncoding] with axis_to_split = 1
    is changed to [2 * Batch, Rows /2, Columns, ValueEncoding].
    The Versions of the dataset are after each other so A B C ->  A_0 A_1 B_0 B_1 C_0 C_1
    :param axis_to_split:
    :param input_encoder_measurement:
    :return:
    """
    list_with_datasets = tf.unstack(
        input_encoder_measurement
    )
    splited_tensors = []
    for tensor in list_with_datasets:
        t_ =sorted_split(tensor)

        #t_ = random_split(axis_to_split, tensor)
        splited_tensors.append(t_)
    input_encoder_measurement_contrastive_loss = tf.concat(splited_tensors, axis=0)

    # old_shape = input_encoder_measurement.shape
    # new_shape = np.array(old_shape)
    # new_shape[axis_to_split] = old_shape[axis_to_split] / 2
    # new_shape[0] = old_shape[0] * 2
    # input_encoder_measurement_contrastive_loss =  tf.reshape(input_encoder_measurement, shape=new_shape)
    return input_encoder_measurement_contrastive_loss


def random_split(axis_to_split, tensor):
    t = tf.split(
        tensor, num_or_size_splits=2, axis=axis_to_split - 1, num=None, name='split'
    )
    t_ = tf.stack(t, axis=0)
    return t_


def sorted_split(tensor):
    arg_sort_y = tf.argsort(tensor[:, -1])
    (ind_0, ind_1) = tf.split(arg_sort_y, num_or_size_splits=2)
    t_0 = tf.gather(tensor, indices=ind_0, axis=0)
    t_1 = tf.gather(tensor, indices=ind_1, axis=0)
    t_ = tf.stack([t_0, t_1], axis=0)
    return t_


def postprocess_contrastive_loss( output_encoder_measurement):
    """
    sums up rows n and n+1 to get n/2 rows
    :param output_encoder_measurement:
    :return:
    """
    old_shape = tf.shape(output_encoder_measurement)
    new_shape = [int(old_shape[0]//2), 2,old_shape[1] ]

    reshaped_tensor = tf.reshape(output_encoder_measurement, new_shape)

    # Sum along the second axis (axis=1)
    summed_tensor = tf.reduce_sum(reshaped_tensor, axis=1)
    mean_tensor = summed_tensor / 2
    return mean_tensor