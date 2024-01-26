import tensorflow as tf
class ReshapeToFlat(tf.Module):
    """Reshapes a tensor of shape (N, D, E) to (1, N, D*E)."""

    def __init__(self, i):
        super(ReshapeToFlat, self).__init__()
        self.i = i
    def __call__(self, X, **kwargs):
        X_reshape = tf.reshape(
            X,
            (X.shape[0], 1, X.shape[1], -1),
            name=f"ReshapeToFlat{self.i}"
        )
        return X_reshape


class ReshapeToNested(tf.Module):
    """Reshapes a tensor of shape (1, N, D*E) to (N, D, E)."""

    def __init__(self, D, i):
        super(ReshapeToNested, self).__init__()
        self.D = D
        self.i = i
    def __call__(self, X,**kwargs):
        X_reshape = tf.reshape(X,
                               (X.shape[0], X.shape[2], self.D, -1),
                               name=f"ReshapeToNested{self.i}"
                               )
        return X_reshape