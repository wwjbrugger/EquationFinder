from typing import Generator

import numpy as np
import tensorflow as tf
import tensorflow_models as tfm

from src.neural_nets.data_set_encoder.measurement_encoder_dummy import MeasurementEncoderDummy

default_kwargs = {
    'num_blocks_text_transformer': 4,
    'float_precision': 3,  # num decimal places
    'mantissa_len': 1,  # num blocks
    'max_exponent': 100,
    'units': 32,
    'vocab_file': "data/text_transformer_vocab.txt",
    'max_dimensions': 3,
    'embedding_dim': 512,
}


def chunks(lst: list, n: int) -> Generator:
    """
    Yield successive n-sized chunks from lst.
    Adapted from https://github.com/facebookresearch/symbolicregression/blob/main/symbolicregression/envs/encoders.py
    """
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


class TextTransformer(MeasurementEncoderDummy):

    def __init__(self, *args, **kwargs):
        super(TextTransformer, self).__init__(*args, **kwargs)
        # get your arguments from kwargs

        kwargs = default_kwargs  # only needed during development

        self.num_blocks_text_transformer = kwargs['num_blocks_text_transformer']
        self.model = self.build_model(args, kwargs)

        # Create the vocabulary
        # Adapted from https://github.com/facebookresearch/symbolicregression/blob/main/symbolicregression/envs/encoders.py
        self.float_precision = kwargs['float_precision']
        self.mantissa_len = kwargs['mantissa_len']
        self.max_exponent = kwargs['max_exponent']
        self.base = (self.float_precision + 1) // self.mantissa_len
        self.max_token = 10 ** self.base
        self.vocab = ["+", "-"]
        self.vocab.extend(
            ["N" + f"%0{self.base}d" % i for i in range(self.max_token)]
        )
        self.vocab.extend(
            ["E" + str(i) for i in range(-self.max_exponent, self.max_exponent + 1)]
        )

        # Define the model layers
        max_dimensions = kwargs['max_dimensions']  # or max_variables
        embedding_dim = kwargs['embedding_dim']
        units = kwargs['units']

        self.lookup = tf.keras.layers.StringLookup(num_oov_indices=0, vocabulary=self.vocab)

        input_length = (self.mantissa_len + 2) * max_dimensions
        self.embedding = tf.keras.layers.Embedding(len(self.vocab), embedding_dim, mask_zero=False,
                                                   input_length=input_length)  # do we need to use mask_zero=True?

        return  # TODO continue from here

        # element-wise!
        self.dense_1 = tf.keras.layers.Dense(
            units,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs
        )

        self.dense_2 = tf.keras.layers.Dense(
            units,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs
        )

        self.encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            activation='relu',
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            use_bias=False,
            norm_first=True,
            norm_epsilon=1e-06,
            intermediate_dropout=0.0,
            **kwargs
        )

    def prepare_data(self, data: dict) -> tf.Tensor:
        # convert data to a tensor
        tensor = tf.convert_to_tensor(data['data_frame'], dtype=tf.float32)
        # add batch dimension
        tensor = tf.expand_dims(tensor, axis=0)
        return tensor

    def call(self, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        tokens = self.encode(x)
        indices = self.lookup(tokens)

        return indices  # TODO continue from here

        inputs = x
        indices = self.lookup(inputs)
        embeddings = self.embedding(indices)
        x1 = self.dense1(embeddings)
        x2 = self.dense2(x1)
        output = self.encoder(x2)

        norm = tf.linalg.norm(output, ord='euclidean', name=None, keepdims=True, axis=-1)
        y = output / norm

        return y

    def build_model(self, *args, **kwargs):
        model = 'Your architecture'
        return model

    def encode(self, values: np.ndarray | tf.Tensor) -> list[str] | list[list[str]]:
        """
        Write a float number.
        Adapted from https://github.com/facebookresearch/symbolicregression/blob/main/symbolicregression/envs/encoders.py
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
        Adapted from https://github.com/facebookresearch/symbolicregression/blob/main/symbolicregression/envs/encoders.py
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
    tt = TextTransformer(**default_kwargs)

    values = np.array([3.745401188473625, 0.3142918568673425, 70.31403])
    print(f"{values=}")
    encoded = tt.encode(values)
    print(f"{encoded=}")
    decoded = tt.decode(encoded)
    print(f"{decoded=}")
    print()
