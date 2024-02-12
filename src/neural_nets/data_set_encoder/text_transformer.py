import tensorflow as tf
import tensorflow_models as tfm

from src.neural_nets.data_set_encoder.measurement_encoder_dummy import MeasurementEncoderDummy


class TextTransformer(MeasurementEncoderDummy):

    def __init__(self, *args, **kwargs):
        super(TextTransformer, self).__init__(*args, **kwargs)
        # get your arguments from kwargs 
        self.num_blocks_text_transformer = kwargs['num_blocks_text_transformer']
        self.model = self.build_model(args, kwargs)

    def prepare_data(self, data):
        norm_frame = self.normalize(
            data_frame=data['data_frame'],
            approach=self.kwargs['normalize_approach']
        )
        # tokenizer here
        tensor_one_row = tf.reshape(tf.convert_to_tensor(norm_frame, dtype=tf.float32), -1)
        # add batch dimension
        tensor_one_row = tf.expand_dims(tensor_one_row, axis=0)
        return tensor_one_row

    def call(self, x, *args, **kwargs):
        # TODO add call
        X_flat_2 = x
        norm = tf.linalg.norm(X_flat_2, ord='euclidean', name=None, keepdims=True, axis=-1)
        out = X_flat_2 / norm

        inputs = out
        indices = self.lookup(inputs)
        embeddings = self.embedding(indices)
        x1 = self.dense1(embeddings)
        x2 = self.dense2(x1)
        output = self.encoder(x2)

        return output

    def build_model(self, *args, **kwargs):
        input_dim = 32
        output_dim = 32
        units = 32

        self.lookup = tf.keras.layers.StringLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token=None,
            oov_token='[UNK]',
            vocabulary=None,
            idf_weights=None,
            encoding=None,
            invert=False,
            output_mode='int',
            sparse=False,
            pad_to_max_tokens=False,
            **kwargs
        )

        self.embedding = tf.keras.layers.Embedding(
            input_dim,
            output_dim,
            embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,
            input_length=None,
            **kwargs
        )

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

        model = 'Your architecture'
        return model
