import tensorflow as tf
from itertools import cycle
from src.neural_nets.data_set_encoder.dataset_modules import MHSA
from sklearn.preprocessing import normalize
import numpy as np
from src.neural_nets.data_set_encoder.measurement_encoder_dummy import MeasurementEncoderDummy
from src.neural_nets.data_set_encoder.reshape import ReshapeToFlat, ReshapeToNested


# based on  https://github.com/OATML/non-parametric-transformers Kossen et al., “Self-Attention Between Datapoints.” and
# https://github.com/arrigonialberto86/set_transformer a tensorflow implementation of Zaheer et al., “Deep Sets.”
class DatasetTransformer(MeasurementEncoderDummy):
    """Non-Parametric Transformers.

       Applies Multi-Head Self-Attention blocks between datapoints,
       and within each datapoint.

       For all model variants, we expect a list of input data, `X_ragged`:
       ```
           len(X_ragged) == N
           X_ragged[i].shape == (D, H_i)
       ```
       In other words, we have `N` input samples. All samples share the same
       number of `D` features, where each feature i is encoded in `H_i`
       dimensions. "Encoding" here refers to the data preprocessing, i.e. the
       one-hot-encoding for categorical features, as well as well as adding
       the mask tokens. (Note that this is done by the code and the user is
       expected to provide datasets as given in `npt.data_loaders`.)

       High-level model overview:

       Initially in NPTModel, `self.in_embedding()` linearly embeds each of the
       `D` feature columns to a shared embedding dimension `E`.
       We learn separate embedding weights for each column.
       This allows us to obtain the embedded data matrix `X_emb` as a
       three-dimensional tensor of shape `(N, D, E)`.
       `E` is referred to as `dim_feat` in the code below.

       After embedding the data, we apply NPT.
       See `build_npt()` for further information.
       NPT applies a series of attention blocks on the input.

       We eventually obtain output of shape `(N, D, E)`,
       which is projected back to the dimensions of the input `X_ragged` using
       `self.out_embedding()`, which applies a learned linear embedding to
       each column `D` separately.
       """

    def __init__(self, *args, **kwargs):
        """Initialise NPTModel.

        Args:
            kwargs: Dict, from which we retrieve:
                input_feature_dims: List[int], used to specify the number of
                    one-hot encoded dimensions for each feature in the table
                    (used when reloading models from checkpoints).
                wandb config
                device: Optional[int].
        """
        super(DatasetTransformer, self).__init__(*args, **kwargs)
        self.kwargs = kwargs
        # * Dataset Metadata *
        # HOw many dimension each feature has
        self.input_feature_dims = kwargs['input_feature_dims']
        # * Dimensionality Configs *
        # how many attention blocks are stacked after each other
        self.stacking_depth = kwargs['stacking_depth']
        # the shared embedding dimension of each attribute is given by
        self.model_dim_hidden = kwargs['model_dim_hidden']
        # how many feature columns are in the input dat
        self.num_input_features = len(self.input_feature_dims)

        self.use_feature_index_embedding = kwargs['use_feature_index_embedding']

        if self.use_feature_index_embedding:
            self.feature_indices = tf.range(
                start=0,
                limit=self.num_input_features,
                delta=1
            )
            self.feature_index_embedding = tf.keras.layers.Embedding(
                input_dim=self.num_input_features,
                output_dim=self.model_dim_hidden
            )
        else:
            self.feature_index_embedding = None

        self.build_model()

    def build_model(self):
        # We immediately embed each element
        # (i.e., a table with N rows and D columns has N x D elements)
        # to the hidden_dim. Similarly, in the output, we will "de-embed"
        # from this hidden_dim.
        # Build encoder
        self.enc = self.get_npt()

        # Hidden dropout is applied for in- and out-embedding
        self.embedding_dropout = (
            tf.keras.layers.Dropout(
                rate=self.kwargs['model_hidden_dropout_prob'],
                name='Embedding_Dropout'

            )
            if self.kwargs['model_hidden_dropout_prob'] else None)

        # *** Input In/Out Embeddings ***
        # Don't use for Image Patching - those are handled by the respective
        # init_image_patching

        # In-Embedding
        # Linearly embeds each of the `D` [len(input_feature_dims)] feature
        # columns to a shared embedding dimension E [model_dim_hidden].
        # Before the embedding, each column has its own dimensionionality
        # H_j [dim_feature_encoding], given by the encoding dimension of the
        # feature (e.g. This is given by the one-hot-encoding size for
        # categorical variables + one dimension for the mask token and two-
        # dimensional for continuous variables (scalar + mask_token)).
        # See docstring of NPTModel for further context.

        self.in_embedding = [
            tf.keras.layers.Dense(units=self.model_dim_hidden, name=f'Embedding_Dense_{i}')
            for i, dim_feature_encoding in enumerate(self.input_feature_dims)]

        # Out embedding.
        # The outputs of the AttentionBlocks have shape (N, D, E)
        # [N, len(input_feature_dim), model_dim_hidden].
        # For each of the column j, we then project back to the dimensionality
        # of that column in the input (N, H_j-1), subtracting 1, because we do
        # not predict the mask tokens, which were present in the input.
        if self.kwargs['use_latent_vector']:
            self.last_layer_0 = tf.keras.layers.Dense(units=64, activation='gelu', name='Dense_Latent_vector_0')
            self.last_layer_1 = tf.keras.layers.Dense(units=32, activation='gelu', name='Dense_Latent_vector_1')
            self.last_layer_2 = tf.keras.layers.Dense(units=14, name='Dense_Latent_vector_2')
        else:
            self.out_embedding = [
                tf.keras.layers.Dense(dim_feature_encoding)  # , activation='relu')
                for dim_feature_encoding in self.input_feature_dims]

    def get_npt(self):
        """
        A model performing "flattened" attention over the rows and
        "nested" attention over the columns.

        This is reasonable if we don't aim to maintain column equivariance
        (which we essentially never do, because of the column-specific
        feature embeddings at the input and output of the NPT encoder).

        This is done by concatenating the feature outputs of column
        attention and inputting them to row attention. Therefore, it requires
        reshaping between each block, splitting, and concatenation.
        """
        if self.stacking_depth < 2:
            raise ValueError(
                f'Stacking depth {self.stacking_depth} invalid.'
                f'Minimum stacking depth is 2.')
        if self.stacking_depth % 2 != 0:
            raise ValueError('Please provide an even stacking depth.')

        print('Building NPT.')

        # *** Construct arguments for row and column attention. ***

        row_att_args = {'kwargs': self.kwargs}
        col_att_args = {'kwargs': self.kwargs}

        # Perform attention over rows first
        att_args = cycle([row_att_args, col_att_args])
        AttentionBlocks = cycle([MHSA])

        D = self.num_input_features

        enc = []

        # Reshape to flattened representation (1, N, D*dim_input)
        enc.append(ReshapeToFlat(i=0))

        enc = self.build_hybrid_enc(
            enc, AttentionBlocks, att_args, D
        )
        return enc

    def build_hybrid_enc(self, enc, AttentionBlocks, att_args, D):
        final_shape = None

        stack = []

        layer_index = 0

        while layer_index < self.stacking_depth:
            if layer_index % 2 == 1:
                # Input is already in nested shape (N, D, E)
                stack.append(
                    next(AttentionBlocks)(
                        dim_in=self.model_dim_hidden,
                        dim_emb=self.model_dim_hidden,
                        dim_out=self.model_dim_hidden,
                        kwargs=next(att_args)['kwargs']
                    )
                )

                # Reshape to flattened representation
                stack.append(ReshapeToFlat(i=layer_index))
                final_shape = 'flat'

            else:
                # Input is already in flattened shape (1, N, D*E)

                # Attend between instances N
                # whenever we attend over the instances,
                # we consider model_dim_hidden = self.c.model_dim_hidden * D
                stack.append(
                    next(AttentionBlocks)(
                        dim_in=self.model_dim_hidden * D,
                        dim_emb=self.model_dim_hidden * D,
                        dim_out=self.model_dim_hidden * D,
                        kwargs=next(att_args)['kwargs']
                    )
                )

                # Reshape to nested representation
                stack.append(ReshapeToNested(D=D, i=layer_index))
                final_shape = 'nested'

            # Conglomerate the stack into the encoder thus far
            enc += stack
            stack = []

            layer_index += 1

        # Reshape to nested representation, for correct treatment
        # after enc
        if final_shape == 'flat':
            enc.append(ReshapeToNested(D=D, i='after_enc'))

        return enc

    def prepare_data(self, x):
        x_old = x
        x = tf.transpose(x, perm=[0, 2, 1])
        # Dataset transformer expect each cell in table to be encoded.
        # we are not doing so. so we add an extra dimension at the end
        tensor = tf.expand_dims(x, axis=-1)
        return tensor

    def call(self, x, *args, **kwargs):
        # x should have  the form list with number columns elements.
        # Each List has # samples elements
        # Each Element has categorical data [one hot, maskeddim], floating [value, masking ]
        # if masked set value to 0 and mask set to 1

        in_dims = [x.shape[0], x.shape[2], x.shape[1], -1]
        x = self.prepare_data(x)
        # encode ragged input array D x {(NxH_j)}_j to NxDxE)
        #  D number features in dataset
        # N number input samples
        # H_J   number embedding of features cn be something like value, mask or power, value/power, mask be,
        # E shared embeding dim
        X_embed = []
        for i, embed in enumerate(self.in_embedding):
            # spaltenweises embedding
            t = x[:, i]
            t_ = embed(t, training=kwargs['training'])
            X_embed.append(t_)
        X_embed = tf.stack(X_embed, axis=2)

        # Compute feature index embeddings, and add them
        if self.feature_index_embedding is not None:
            feature_index_embeddings = self.feature_index_embedding(
                self.feature_indices)

            # Add a batch dimension (the rows)
            feature_index_embeddings = tf.expand_dims(
                input=feature_index_embeddings,
                axis=0
            )
            # Add a batch dimension (the batches)
            feature_index_embeddings = tf.expand_dims(
                input=feature_index_embeddings,
                axis=0
            )

            # Repeats this tensor along the specified dimensions.            #
            # Unlike expand(), this function copies the tensor’s data.
            feature_index_embeddings = tf.repeat(input=feature_index_embeddings,
                                                 repeats=X_embed.shape[1],
                                                 axis=1)
            feature_index_embeddings = tf.repeat(input=feature_index_embeddings,
                                                 repeats=X_embed.shape[0],
                                                 axis=0)

            # Add to X
            X_embed = X_embed + feature_index_embeddings

        # Embedding tensor currently has shape (B x N x D x E)

        X_embed = X_embed

        if self.embedding_dropout is not None:
            X_embed = self.embedding_dropout(X_embed, training=kwargs['training'])

        # apply NPT
        X_enc = X_embed
        for layer in self.enc:
            X_enc = layer(X_enc, training=kwargs['training'])

        if X_enc.shape[2] == in_dims[1]:
            # for uneven stacking_depth, need to permute one last time
            # to obtain output of shape (N, D, E)
            X_enc = tf.transpose(X_enc, perm=(0, 2, 1, 3))

        # Dropout before final projection (follows BERT, which performs
        # dropout before e.g. projecting to logits for sentence classification)
        if self.embedding_dropout is not None:
            X_enc = self.embedding_dropout(X_enc, training=kwargs['training'])

        if self.kwargs['use_latent_vector']:
            X_flat = tf.reshape(X_enc, (X_enc.shape[0], -1))
            X_flat_0 = self.last_layer_0(X_flat)
            X_flat_1 = self.last_layer_1(X_flat_0)
            X_flat_2 = self.last_layer_2(X_flat_1)
            norm = tf.linalg.norm(X_flat_2, ord='euclidean', name=None, keepdims=True, axis=-1)
            out = X_flat_2 / norm

            # x_out = self.latent_measurement_norm(x_out)
            return out
        else:
            # project back to ragged (dimensions D x {(NxH_j)}_j )
            # Is already split up across D
            x_out = [de_embed(X_enc[:, :, i], training=kwargs['training'])
                     for i, de_embed in enumerate(self.out_embedding)]

            x_out = tf.stack(x_out, axis=2)

            return tf.squeeze(x_out)
