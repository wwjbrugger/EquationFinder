import tensorflow as tf
import numpy as np

def expand_tensor_to_same_size(to_change, reference):
    if to_change.ndim < reference.ndim:
        how_often_to_expand =  reference.ndim - to_change.ndim
        for i in range(how_often_to_expand):
            to_change = tf.expand_dims(to_change, axis=0)
    return to_change

def check_for_non_numeric_and_replace_by_0(logger, tensor, name):
    try:
        tf.debugging.check_numerics(tensor, message=f'Checking {name}]')
    except Exception as e:
        logger.error(f'Checking {name}: {tensor}')
        tensor = tf.where(
            tf.math.is_finite(tensor),
            x=tensor,
            y=tf.constant(0.0, dtype=tf.float32)
        )
    return tensor

def cast_float32_to_bit_representation(value):
    int32bits = np.float32(value).view(np.uint32).item()
    boolean_str = f'{int32bits:032b}'
    bit_array = np.array([bit for bit in boolean_str], dtype=np.float32)
    return bit_array

