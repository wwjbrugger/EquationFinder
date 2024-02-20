from typing import Generator

import numpy as np
import tensorflow as tf
import tensorflow_models as tfm

from src.neural_nets.data_set_encoder.measurement_encoder_dummy import MeasurementEncoderDummy


def chunks(lst: list, n: int) -> Generator:
    """
    Yield successive n-sized chunks from lst.
    (Adapted from Kamienny et al.)
    """
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


class TextTransformer(MeasurementEncoderDummy):
    """
    Basically a reimplementation of the tokenizer, embedder, and encoder described by Kamienny et al. in "End-to-end
    symbolic regression with transformers" (2022). Parts of the code are directly adapted from their code at
    https://github.com/facebookresearch/symbolicregression/blob/main/symbolicregression/envs/encoders.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get your arguments from kwargs

        self.float_precision = kwargs['float_precision']
        self.mantissa_len = kwargs['mantissa_len']
        self.max_exponent = kwargs['max_exponent']
        self.max_dimensions = kwargs['max_dimensions']
        self.embedding_dim = kwargs['embedding_dim']
        self.embedder_intermediate_expansion_factor = kwargs['embedder_intermediate_expansion_factor']
        self.num_encoder_layers = kwargs['num_encoder_layers']
        self.num_attention_heads = kwargs['num_attention_heads']
        self.encoder_intermediate_size = kwargs['encoder_intermediate_size']

        # Create the vocabulary (adapted from Kamienny et al.)

        self.base = (self.float_precision + 1) // self.mantissa_len
        self.max_token = 10 ** self.base
        self.vocab = ["+", "-"]
        self.vocab.extend(["N" + f"%0{self.base}d" % i for i in range(self.max_token)])
        self.vocab.extend(["E" + str(i) for i in range(-self.max_exponent, self.max_exponent + 1)])

        # Define the model layers
        # TODO use dropout somewhere?

        self.lookup = tf.keras.layers.StringLookup(num_oov_indices=0, vocabulary=self.vocab)

        self.input_length = (self.mantissa_len + 2) * self.max_dimensions
        self.embedding = tf.keras.layers.Embedding(len(self.vocab), self.embedding_dim, mask_zero=False,
                                                   input_length=self.input_length)  # do we need to use mask_zero=True?

        self.embedder_intermediate_size = self.input_length * self.embedding_dim * self.embedder_intermediate_expansion_factor
        self.dense_1 = tf.keras.layers.Dense(self.embedder_intermediate_size, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(self.embedding_dim, activation='relu')

        self.encoder = tfm.nlp.models.TransformerEncoder(num_layers=self.num_encoder_layers,
                                                         num_attention_heads=self.num_attention_heads,
                                                         intermediate_size=self.encoder_intermediate_size)  # should we use attention bias?

    def prepare_data(self, data: dict) -> tf.Tensor:
        """
        Prepare the data for the model. For this architecture, we skip the normalization and only unpack the data and
        add a batch dimension.
        :param data: Dictionary containing the measurement data in a dataframe
        """
        tensor = tf.convert_to_tensor(data['data_frame'], dtype=tf.float32)
        tensor = tf.expand_dims(tensor, axis=0)
        return tensor

    def call(self, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Propagate the input data forward through the model to compute a single embedding for the measurements.
        :param x: Input data tensor
        """
        tokens = self.encode(x)
        indices = self.lookup(tokens, **kwargs)
        embeddings = self.embedding(indices, **kwargs)
        embeddings_concatenated = tf.reshape(embeddings, [*embeddings.shape[0:-2], -1])  # or tf.keras.layers.Reshape?
        projected_down_1 = self.dense_1(embeddings_concatenated, **kwargs)
        projected_down_2 = self.dense_2(projected_down_1, **kwargs)
        y = self.encoder(projected_down_2, **kwargs)
        y_pooled = tf.math.reduce_max(y, axis=-2)  # or tf.keras.layers.GlobalMaxPool1D?
        y_normalized, _ = tf.linalg.normalize(y_pooled, ord='euclidean', axis=-1)  # or tf.keras.layers.Normalization?
        return y_normalized

    def encode(self, values: np.ndarray | tf.Tensor) -> list[str] | list[list[str]]:
        """
        Write a float number.
        (Adapted from Kamienny et al.)
        """
        if len(values.shape) == 1:
            seq = []
            for val in values:
                assert val not in [-np.inf, np.inf]
                sign = "+" if val >= 0 else "-"
                m, e = (f"%.{self.float_precision}e" % val).split("e")
                i, f = m.lstrip("-").split(".")
                i = i + f
                tokens = chunks(i, self.base)
                expon = int(e) - self.float_precision
                if expon < -self.max_exponent:
                    tokens = ["0" * self.base] * self.mantissa_len
                    expon = int(0)
                seq.extend([sign, *["N" + token for token in tokens], "E" + str(expon)])
            return seq
        else:
            seqs = [self.encode(values[0])]
            N = values.shape[0]
            for n in range(1, N):
                seqs += [self.encode(values[n])]
        return seqs

    def decode(self, lst: list[str]) -> None | float | list[float]:
        """
        Parse a list that starts with a float.
        Return the float value, and the position it ends in the list.
        (Adapted from Kamienny et al.)
        """
        if len(lst) == 0:
            return None
        seq = []
        for val in chunks(lst, 2 + self.mantissa_len):
            for x in val:
                if x[0] not in ["-", "+", "E", "N"]:
                    return np.nan
            try:
                sign = 1 if val[0] == "+" else -1
                mant = ""
                for x in val[1:-1]:
                    mant += x[1:]
                mant = int(mant)
                exp = int(val[-1][1:])
                value = sign * mant * (10 ** exp)
                value = float(value)
            except Exception:
                value = np.nan
            seq.append(value)
        return seq


if __name__ == "__main__":
    # Only for debugging
    default_kwargs = {
        'float_precision': 3,  # num decimal places
        'mantissa_len': 1,  # num blocks
        'max_exponent': 100,
        'max_dimensions': 3,  # max variables in measurement data (x_1, ..., x_n, y)
        'embedding_dim': 512,
        'embedder_intermediate_expansion_factor': 1,
        'num_encoder_layers': 4,
        'num_attention_heads': 8,
        'encoder_intermediate_size': 2048,
    }
    tt = TextTransformer(**default_kwargs)

    values = np.array([3.745401188473625, 0.3142918568673425, 70.31403])
    print(f"{values=}")
    encoded = tt.encode(values)
    print(f"{encoded=}")
    decoded = tt.decode(encoded)
    print(f"{decoded=}")
    print()
