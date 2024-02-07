import tensorflow as tf
import numpy as np

def categorical_crossentropy_loss(real, pred):
    cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none')
    loss = cross_entropy(y_true=real, y_pred=pred)
    loss = tf.reduce_mean(loss)
    return loss


def kl_divergence(real, pred):
    kl = tf.keras.losses.KLDivergence()
    loss = kl(real, pred)
    return loss


def mean_square_error_loss_function(real, pred):
    mean_square = tf.keras.losses.MeanSquaredError(name='mean_squared_error')
    loss = mean_square(y_true=real, y_pred=pred)
    return loss


def mean_abs_error(real, pred):
    mean_abs = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
    loss = mean_abs(y_true=real, y_pred=pred)
    return loss


### This is the SimCLRLoss from kerasCV.
# To not add another dependency the calls to kerascv funktion are replaced by its tf equivalents

LARGE_NUM = 1e9


def l2_normalize(x, axis):
    epsilon = tf.keras.backend.epsilon()
    power_sum = tf.math.reduce_sum(tf.math.square(x), axis=axis, keepdims=True)
    norm = tf.math.reciprocal(tf.math.sqrt(tf.math.maximum(power_sum, epsilon)))
    return tf.math.multiply(x, norm)


def shape(x):
    """Always return a tuple shape.

    `tf.shape` will return a `tf.Tensor`, which differs from the tuple return
    type on the torch and jax backends. We write our own method instead which
    always returns a tuple, with integer values when the shape is known, and
    tensor values when the shape is unknown (this is tf specific, as dynamic
    shapes do not apply in other backends).
    """
    if not tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    dynamic = tf.shape(x)
    if x.shape == tf.TensorShape(None):
        raise ValueError(
            "All tensors passed to `ops.shape` must have a statically known "
            f"rank. Received: x={x} with unknown rank."
        )
    static = x.shape.as_list()
    return tuple(dynamic[i] if s is None else s for i, s in enumerate(static))



class NT_Xent(tf.keras.layers.Layer):
    """ Normalized temperature-scaled CrossEntropy loss [1]
        [1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” arXiv. 2020, Accessed: Jan. 15, 2021. [Online]. Available: https://github.com/google-research/simclr.
        From https://github.com/gabriel-vanzandycke/tf_layers/blob/main/tf_layers/layers.py
          """
    def __init__(self, args, **kwargs):
        self.args = args
        super().__init__(**kwargs)
        # closer to -1 indicate greater similarity, 0 indicates orthogonality, The values closer to 1 indicate greater dissimilarity
        self.similarity = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
        #self.criterion = tf.keras.losses.MeanSquaredError()
        self.criterion = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.mask_for_same_dataset = np.zeros(
            shape=(self.args.batch_size_training*2, self.args.batch_size_training*2),
            dtype=np.bool_
        )
        for i in range(self.args.batch_size_training):
            self.mask_for_same_dataset[2 * i, 2 * i + 1] = True

        self.mask_contrast_dataset = np.triu(np.ones(
            shape=(self.args.batch_size_training*2, self.args.batch_size_training*2),
            dtype=np.bool_), k=1)
        self.mask_contrast_dataset[self.mask_for_same_dataset] = False


    def __call__(self, zizj, target_dataset_encoding=None, tau=None):
        """ zizj is [B,N] tensor with order z_i1 z_j1 z_i2 z_j2 z_i3 z_j3 ...
            batch_size is twice the original batch_size
        """
        if not target_dataset_encoding is None:
            sim = -1 * self.similarity(tf.expand_dims(zizj, 1), tf.expand_dims(zizj, 0))
            sim_activation = (sim + 1) / 2

            contrast_loss = self.criterion(
                target_dataset_encoding[self.mask_contrast_dataset],
                sim_activation[self.mask_contrast_dataset]
            )
            similarity_loss = self.criterion(
                target_dataset_encoding[self.mask_for_same_dataset],
                sim_activation[self.mask_for_same_dataset]
            )

            sim_loss = similarity_loss + 0.1 * contrast_loss
            return sim_loss


