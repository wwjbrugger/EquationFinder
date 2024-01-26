"""Contains base attention modules."""

import math

import tensorflow as tf


class MHSA(tf.keras.Model):
    """
    Multi-head Self-Attention Block.

    Based on implementation from Set Transformer (Lee et al. 2019,
    https://github.com/juho-lee/set_transformer).
    Alterations detailed in MAB method.
    """
    has_inducing_points = False

    def __init__(self, dim_in, dim_emb, dim_out, kwargs):
        super(MHSA, self).__init__()
        self.mab = MAB(dim_Q=dim_in,
                       dim_KV=dim_in,
                       dim_emb=dim_emb,
                       dim_out=dim_out,
                       kwargs=kwargs
                       )

    def call(self, X, *args, **kwargs):
        return self.mab(X, X, training=kwargs['training'])


class MAB(tf.keras.Model):
    """Multi-head Attention Block.

    Based on Set Transformer implementation
    (Lee et al. 2019, https://github.com/juho-lee/set_transformer).
    """

    def __init__(
            self, dim_Q, dim_KV, dim_emb, dim_out, kwargs):
        """

        Inputs have shape (B_A, N_A, F_A), where
        * `B_A` is a batch dimension, along we parallelize computation,
        * `N_A` is the number of samples in each batch, along which we perform
        attention, and
        * `F_A` is dimension of the embedding at input
            * `F_A` is `dim_Q` for query matrix
            * `F_A` is `dim_KV` for key and value matrix.

        Q, K, and V then all get embedded to `dim_emb`.
        `dim_out` is the output dimensionality of the MAB which has shape
        (B_A, N_A, dim_out), this can be different from `dim_KV` due to
        the head_mixing.

        This naming scheme is inherited from set-transformer paper.
        """
        super(MAB, self).__init__()
        self.kwargs = kwargs
        mix_heads = self.kwargs['model_mix_heads']
        self.num_heads = self.kwargs['num_heads']
        sep_res_embed = self.kwargs['model_sep_res_embed']
        ln = self.kwargs['model_att_block_layer_norm']
        rff_depth = self.kwargs['model_rff_depth']
        self.att_score_norm = self.kwargs['model_att_score_norm']
        self.pre_layer_norm = self.kwargs['model_pre_layer_norm']

        if dim_out is None:
            dim_out = dim_emb
        elif (dim_out is not None) and (mix_heads is None):
            print('Warning: dim_out transformation does not apply.')
            dim_out = dim_emb

        self.dim_KV = dim_KV
        self.dim_split = dim_emb // self.num_heads
        self.fc_q = tf.keras.layers.Dense(
            units=dim_emb,
            activation=None,
            name=f"fc_q_dense",
            use_bias=False
        )
        self.fc_k = tf.keras.layers.Dense(
            units=dim_emb,
            activation=None,
            name=f"fc_k_dense",
            use_bias=False
        )
        self.fc_v = tf.keras.layers.Dense(
            units=dim_emb,
            activation=None,
            name=f"fc_v_dense",
            use_bias=False
        )
        self.fc_mix_heads = tf.keras.layers.Dense(
            units=dim_out,
            activation='gelu',
            name="fc_mix_heads_dense"
        ) if mix_heads else None
        self.fc_res = tf.keras.layers.Dense(
            dim_out,
            activation='gelu',
            name=f"res_dense"
        ) if sep_res_embed else None

        if ln:
            # Applied to X
            self.ln0 = tf.keras.layers.LayerNormalization(
                axis=-1,
                epsilon=self.kwargs['model_layer_norm_eps'],
                name="PreLayerNormalization"
            )

            self.ln1 = tf.keras.layers.LayerNormalization(
                axis=-1,
                epsilon=self.kwargs['model_layer_norm_eps'],
                name="PostLayerNormalization"
            )
        else:
            self.ln0 = None
            self.ln1 = None

        self.hidden_dropout = (
            tf.keras.layers.Dropout(
                rate=self.kwargs['model_hidden_dropout_prob'],
                name="Dropout_Hidden"
                if self.kwargs['model_hidden_dropout_prob'] else None)
        )

        self.att_scores_dropout = (
            tf.keras.layers.Dropout(
                rate=self.kwargs['model_att_score_dropout_prob'],
                name="Dropout_Att_Scores"
                if self.kwargs['model_att_score_dropout_prob'] else None)
        )

        self.init_rff(dim_out, rff_depth)

    def init_rff(self, dim_out, rff_depth):
        # Linear layer with 4 times expansion factor as in 'Attention is
        # all you need'!
        self.rff = [
            tf.keras.layers.Dense(
                units=4 * dim_out,
                activation='gelu',
                name=f"DenseRff_{rff_depth}"
            ),
            tf.keras.activations.gelu
        ]

        if self.hidden_dropout is not None:
            self.rff.append(self.hidden_dropout)

        for i in range(rff_depth - 1):
            self.rff += [
                tf.keras.layers.Dense(
                    units=4 * dim_out,
                    activation=None,
                    name=f"DenseRff_{i}"
                ),
                tf.keras.activations.gelu
            ]

            if self.hidden_dropout is not None:
                self.rff.append(self.hidden_dropout)

        self.rff += [tf.keras.layers.Dense(
            units=dim_out,
            activation='gelu',
            name=f"DenseRff_Last",
        )]

        if self.hidden_dropout is not None:
            self.rff.append(self.hidden_dropout)

        # self.rff = tf.Sequential(*self.rff)

    def call(self, X, Y, *args, **kwargs):
        if self.pre_layer_norm and self.ln0 is not None:
            X_multihead = self.ln0(X, training=kwargs['training'])
        else:
            X_multihead = X

        Q = self.fc_q(X_multihead, training=kwargs['training'])

        if self.fc_res is None:
            X_res = Q
        else:
            X_res = self.fc_res(X, training=kwargs['training'])  # Separate embedding for residual

        K = self.fc_k(Y, training=kwargs['training'])
        V = self.fc_v(Y, training=kwargs['training'])

        # zerlegt die Key, Query und Value matrix in mehrere Heads
        Q_ = tf.concat(
            tf.split(value=Q,
                     num_or_size_splits=int(Q.shape[3] / self.dim_split),
                     axis=3),
            1
        )
        K_ = tf.concat(
            tf.split(value=K,
                     num_or_size_splits=int(K.shape[3] / self.dim_split),
                     axis=3),
            1
        )
        V_ = tf.concat(
            tf.split(value=V,
                     num_or_size_splits=int(V.shape[3] / self.dim_split),
                     axis=3),
            1
        )

        # TODO: track issue at
        # https://github.com/juho-lee/set_transformer/issues/8
        # A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        A = tf.einsum('bijl,bikl->bijk', Q_, K_)

        if self.att_score_norm == 'softmax':
            A_norm = tf.nn.softmax(A / math.sqrt(self.dim_KV), axis=3)
        elif self.att_score_norm == 'constant':
            A_norm = A / self.dim_split
        else:
            raise NotImplementedError

        # Attention scores dropout is applied to the N x N_v matrix of
        # attention scores.
        # Hence, it drops out entire rows/cols to attend to.
        # This follows Vaswani et al. 2017 (original Transformer paper).

        if self.att_scores_dropout is not None:
            A_norm = self.att_scores_dropout(A_norm, training=kwargs['training'])

        multihead = tf.matmul(A_norm, V_)
        multihead_split =  tf.split(value=multihead,
                            num_or_size_splits=int(multihead.shape[1] / Q.shape[1]),
                            axis=1)
        multihead_concat = tf.concat(
            values=multihead_split,
            axis=3
        )

        # Add mixing of heads in hidden dim.

        if self.fc_mix_heads is not None:
            H = self.fc_mix_heads(multihead_concat, training=kwargs['training'])
        else:
            H = multihead_concat

        # Follow Vaswani et al. 2017 in applying dropout prior to
        # residual and LayerNorm
        if self.hidden_dropout is not None:
            H = self.hidden_dropout(H, training=kwargs['training'])

        # True to the paper would be to replace
        # self.fc_mix_heads = nn.Linear(dim_V, dim_Q)
        # and Q_out = X
        # Then, the output dim is equal to input dim, just like it's written
        # in the paper. We should definitely check if that boosts performance.
        # This will require changes to downstream structure (since downstream
        # blocks expect input_dim=dim_V and not dim_Q)

        # Residual connection
        Q_out = X_res
        H_after_res = H + Q_out

        # Post Layer-Norm, as in SetTransformer and BERT.
        if not self.pre_layer_norm and self.ln0 is not None:
            H_after_res = self.ln0(H_after_res, training=kwargs['training'])

        if self.pre_layer_norm and self.ln1 is not None:
            H_rff = self.ln1(H_after_res, training=kwargs['training'])
        else:
            H_rff = H_after_res

            # Apply row-wise feed forward network
        for layer in self.rff:
            # todo add training parameter
            H_rff = layer(H_rff)
            expanded_linear_H = H_rff

        # Residual connection
        expanded_linear_H = H_after_res + expanded_linear_H


        if not self.pre_layer_norm and self.ln1 is not None:
            expanded_linear_H = self.ln1(
                expanded_linear_H,
                training= kwargs['training']
            )

        return expanded_linear_H
