import time
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from src.utils.parse_args import str2bool


class Config:
    @classmethod
    def arguments_parser(cls) -> ArgumentParser:
        ### General
        parser = ArgumentParser(description="A MuZero and AlphaZero implementation in Tensorflow.")

        parser.add_argument("--experiment_name", type=str, default='delete_me')
        parser.add_argument("--job_id", type=int,default=0)
        parser.add_argument("--path_to_complete_model", type=str,
                            default='', help="Path to a complete model  which should be loaded")
        parser.add_argument("--path_to_pretrained_dataset_encoder",
                            help='Path to a pretrained model where the dataset encoder is copied from.',
                             type=str)
        parser.add_argument("--run_mcts", type=str2bool,
                            default=True, help="If mcts should be run for training"
                                               "If False only replay buffer is used.")
        parser.add_argument("--only_test", type=str2bool,
                            default=False, help="If only test mode should be run")
        parser.add_argument("--seed", type=int, required=True)
        parser.add_argument("--minutes_to_run", type=int)
        parser.add_argument("--max_iteration_to_run", type=int,
                            default=200)
        parser.add_argument("--logging_level", type=int, default=30,
                            help="CRITICAL = 50, ERROR = 40, "
                                 "WARNING = 30, INFO = 20, "
                                 "DEBUG = 10, NOTSET = 0")
        parser.add_argument("--wandb", type=str, default='offline',
                            choices=["online", "offline", "disabled"])
        parser.add_argument("--gpu", default=0,
                            help="Set which device to use (-1 for CPU). Equivalent "
                                 "to/overrides the CUDA_VISIBLE_DEVICES environment variable.")

        parser.add_argument("--data", dest="data_path", type=Path,
                            help="path to preprocessed dataset", required=False)
        parser.add_argument("--num_selfplay_iterations", type=int, default=10,
                            help="(int) Number of games played in one iteration")
        parser.add_argument("--num_selfplay_iterations_test", type=int, default=2,
                            help="(int) Number of games played in one iteration")
        parser.add_argument("--test_network", type=str2bool, default=True,
                            help="If the network should be tested with a "
                                 "separate network")
        parser.add_argument("--test_every_n_steps", type=int, default=10,
                            help="How many training epochs there should "
                                 "before network is tested")
        parser.add_argument(
            "--max_ram", type=int, default=20,
            help="Maximal allowed RAM allowed to use by process given in GB. "
                 "Conversion from Bytes in GB by 10^9 "
        )

        ### Infos about Tree
        parser.add_argument('--minimum_reward', type=np.float32,
                            default=-1)
        parser.add_argument('--maximum_reward', type=np.float32,
                            default=1)
        parser.add_argument('--max_depth_of_tree', type=int,
                            default=10,
                            help='Maximum depth of generated equations')
        parser.add_argument('--max_num_nodes_in_syntax_tree', type=int,
                            help='Maximum nodes of generated equations', default=25)
        parser.add_argument('--max_branching_factor', type=np.float32,
                            default=2,
                            help='Estimate how many children a node will have at average')
        parser.add_argument('--max_constants_in_tree', type=int,
                            default=3,
                            help='Maximum number of constants allowed in  equation'
                                 'afterwards equation will be invalid')
        ### Training neural net

        parser.add_argument("--batch_size_training", type=int,
                            default=16,
                            help='Batch size used for learning. Depends on available space on GPU or CPU')
        parser.add_argument("--num_gradient_steps", type=int,
                            default=200,
                            help="(int) Number of weight updates to perform in the backpropagation step in one iteration")
        parser.add_argument("--average_policy_if_wrong", type=str2bool,
                            default=False,
                            help='If network found equation which gets minimum reward_should it learn '
                                 'as policy an uniform distribution')
        parser.add_argument("--cold_start_iterations", type=int,
                            default=20,
                            help='How many iterations should be searched without training the network ')

        ## Preprocess
        parser.add_argument("--equation_preprocess_class", type=str,
                            default='PandasPreprocess',
                            choices=['EquationPreprocessDummy',
                                     'PandasPreprocess'],
                            help='Datasets can be represent in multiple ways.'
                                 'EquationPreprocessDummy is a interface and if selected, the '
                                 'system will not use any information from the equation representation.'
                            )
        parser.add_argument(
            "--max_len_datasets", type=int,
            default=20,
            help="Number of samples from dataset which is the input into NN"
        )

        ## Encoder Equations
        parser.add_argument("--class_equation_encoder",
                            default='Transformer_Encoder_String',
                            type=str,
                            choices=['EquationEncoderDummy',
                                     'Transformer_Encoder_String'],
                            help='Equations are represented as syntax trees. '
                                 'Select the class to embed them.')

        parser.add_argument("--embedding_dim_encoder_equation",
                            default=8, type=int,
                            help='Symbols have to be mapped on a float vector how many dimensions to use')
        parser.add_argument("--max_tokens_equation", type=int,
                            default=64,
                            help='How many symbols should the tree representation have. '
                                 'everything above will be truncated, below will be pad')
        parser.add_argument("--use_position_encoding", type=str2bool,
                            default=False,
                            help='Should the node which is going to be expand '
                                 'be add to tree representation ')
        ## Attention String

        parser.add_argument("--num_layer_encoder_equation_transformer", type=int,
                            default=2,
                            help='How many attention layer are stacked ')
        parser.add_argument("--num_heads_encoder_equation_transformer", type=int,
                            default=4,
                            help='Number of attention heads in equation attention mechanism')
        parser.add_argument("--dim_feed_forward_equation_encoder_transformer", type=int,
                            default=32,
                            help='Number of units in Feed-forward-net in attention mechanism ')
        parser.add_argument("--dropout_rate", type=np.float32,
                            default=0.1,
                            help='dropout rate of mlp in attention mechanism ')
        ########## Encoder Measurement
        parser.add_argument("--class_measurement_encoder", type=str,
                            choices=['MeasurementEncoderDummy',
                                     'LSTM_Measurement_Encoder',
                                     'Bi_LSTM_Measurement_Encoder',
                                     'MLP_Measurement_Encoder',
                                     'DatasetTransformer',
                                     'MeasurementEncoderPicture',
                                     'TextTransformer'
                                     ]
                            )
        parser.add_argument("--normalize_approach", type=str,
                            help="How to normalize values:"
                                 "abs_max_y normalize y values to the range -1 and 1"
                                 "lin_transform transforms x values to the range -1 and 1 "
                                 "both option can be used together by writing both keywords."
                            )
        parser.add_argument("--contrastive_loss", type=str2bool,
                            default=True,
                            help="If contrastive loss should be used" )

        ########## Encoder Measurement LSTM
        parser.add_argument("--encoder_measurements_LSTM_units", type=int,
                            default=64,
                            help='Positive integer, dimensionality of the output space.')

        parser.add_argument("--encoder_measurements_LSTM_return_sequence",
                            default=True,
                            type=str2bool,
                           )
        ######### Encoder Measurement  MLP
        parser.add_argument("--encoder_measurement_num_layer", type=int,
                            default=3,
                            help='How many layer the multi layer perceptron should have')
        parser.add_argument("--encoder_measurement_num_neurons", type=int,
                            default=64,
                            help='How many neurons one layer should have')

        ######### Encoder Measurement DatasetTransformer

        parser.add_argument("--model_dim_hidden_dataset_transformer", type=int,
                            default=64,
                            help="The shared embedding dimension of each attribute is given by")
        parser.add_argument("--model_num_heads_dataset_transformer", type=int,
                            default=8,
                            help=" num_heads attention heads")
        parser.add_argument("--model_stacking_depth_dataset_transformer", type=int,
                            default=4,
                            help='How many attention blocks are stacked after each other'
                                 'original paper 8')
        parser.add_argument('--model_sep_res_embed_dataset_transformer',
                          default=True,
                          type=str2bool,
                            help='Use a seperate query embedding W^R to construct the residual '
                                 'connection '
                                 'W^R Q + MultiHead(Q, K, V)'
                                 'in the multi-head attention. This was not done by SetTransformers, '
                                 'which reused the query embedding matrix W^Q, '
                                 'but we feel that adding a separate W^R should only helpful.'
                            )
        parser.add_argument('--model_att_block_layer_norm_dataset_transformer',
                            type=str2bool,
                            default=True,
                            help='(Disable) use of layer normalization in attention blocks.')
        parser.add_argument('--model_layer_norm_eps_dataset_transformer',
                            type=float,
                            default=1e-12 ,
                            help='The epsilon used by layer normalization layers.'
                                 'Default from BERT.')
        parser.add_argument('--model_att_score_norm_dataset_transformer',
                            type=str,
                            default='softmax',
                            help='Normalization to use for the attention scores. Options include'
                                 'softmax, constant (which divides by the sqrt of # of entries).')
        parser.add_argument('--model_pre_layer_norm_dataset_transformer',
                           type=str2bool,
                            default=False,
                            help='If True, we apply the LayerNorm (i) prior to Multi-Head '
                                 'Attention, (ii) before the row-wise feedforward networks. '
                                 'SetTransformer and Huggingface BERT opt for post-LN, in which '
                                 'LN is applied after the residual connections. See `On Layer '
                                 'Normalization in the Transformer Architecture (Xiong et al. '
                                 '2020, https://openreview.net/forum?id=B1x8anVFPr) for '
                                 'discussion.')
        parser.add_argument('--model_rff_depth_dataset_transformer',
                            type=int,
                            default=2,
                            help=f'Number of layers in rFF block.'
                            )

        parser.add_argument('--model_hidden_dropout_prob_dataset_transformer',
                            type=float,
                            default=1e-06,
                            help='The dropout probability for all fully connected layers in the '
                                 '(in, but not out) embeddings, attention blocks.')
        parser.add_argument('--model_att_score_dropout_prob_dataset_transformer',
                            type=float,
                            default=1e-06,
                            help='The dropout ratio for the attention scores.')
        parser.add_argument('--model_mix_heads_dataset_transformer',
                            type=str2bool,
                            default=True,
                            help=f'Add linear mixing operation after concatenating the outputs '
                                 f'from each of the heads in multi-head attention.'
                                 f'Set Transformer does not do this. '
                                 f'We feel that this should only help. But it also does not really '
                                 f'matter as the rFF(X) can mix the columns of the multihead attention '
                                 f'output as well. '
                                 f'model_mix_heads=False may lead to inconsistencies for dimensions.')
        parser.add_argument('--model_embedding_layer_norm_dataset_transformer',
                          type=str2bool,
                            default=False,
                            help='(Disable) use of layer normalization after in-/out-embedding.')

        parser.add_argument('--dataset_transformer_use_latent_vector',
                            type=str2bool,
                            default=True,
                            help="If the output of the dataset transformer should have the shape of the "
                                 "dataset (for testing) or be an Tensor with the shape (batch, -1 ) "
                            )
        parser.add_argument('--bit_embedding_dataset_transformer',
                            type=str2bool,
                            default=False,
                            help="If bit embedding should be used in dataset transformer"
                                 "if true the float is transformed into an array of 32 1.0|0.0"
                                 "if false the float are feed into the ned as they are"
                            )
        parser.add_argument('--use_feature_index_embedding_dataset_transformer',
                            default=False,
                            type=str2bool,
                            help="Add a column depending  embedding to the dataset representation")

        ######### Encoder Measurement TextTransformer
        parser.add_argument('--float_precision_text_transformer', default=3, type=int,
                            help="Number of decimal places for the token representation of the measurement data. "
                                 "The number of digits in the mantissa is float_precision + 1.")
        parser.add_argument('--mantissa_len_text_transformer', default=1, type=int,
                            help="Number of mantissa blocks in the token representation of the measurement data.")
        parser.add_argument('--max_exponent_text_transformer', default=100, type=int,
                            help="Maximum (and negative minimum) exponent in the token representation of the measurement data.")
        parser.add_argument('--num_dimensions_text_transformer', default=3, type=int,
                            help="Exact number of variables (x_1, ..., x_n, y) of the measurement data.")
        parser.add_argument('--embedding_dim_text_transformer', default=512, type=int,
                            help="Size of the embedding vectors.")
        parser.add_argument('--embedder_intermediate_expansion_factor_text_transformer', default=1.0, type=float,
                            help="By how much to increase (or reduce) the embedding size from the first to the second dense layer in the embedder.")
        parser.add_argument('--num_encoder_layers_text_transformer', default=4, type=int,
                            help="Number of encoder layers in the text transformer.")
        parser.add_argument('--num_attention_heads_text_transformer', default=8, type=int,
                            help="Number of attention heads per layer in the text transformer.")
        parser.add_argument('--encoder_intermediate_expansion_factor_text_transformer', default=4.0, type=float,
                            help="By how much to increase (or reduce) the embedding size from the first to the second dense layer in the encoder.")
        parser.add_argument('--intermediate_dropout_rate_text_transformer', default=0.2, type=float,
                            help="Dropout probability for all places except the attention layer.")
        parser.add_argument('--attention_dropout_rate_text_transformer', default=0.1, type=float,
                            help="Dropout probability for the attention layer.")

        ## Actor Decoder
        parser.add_argument("--actor_decoder_class", type=str,
                            default='mlp_decoder',
                            choices=['mlp_decoder'])
        parser.add_argument('--actor_decoder_normalize_way', type=str,
                            default='soft_max',
                            choices=['soft_max', 'positive_sum_1', 'sigmoid', 'None', 'tanh'],
                            help='Which normalization should be used at the end of the actor'
                            )
        ## Critic Decoder
        parser.add_argument("--critic_decoder_class", type=str,
                            default='mlp_decoder',
                            choices=['mlp_decoder'])
        parser.add_argument('--critic_decoder_normalize_way', type=str,
                            default= 'tanh',
                            choices=['soft_max', 'positive_sum_1', 'sigmoid', 'None', 'tanh'],
                            help='Which normalization should be used at the end of the actor'
                            )
        ## MCTS
        parser.add_argument('--MCTS_engine', type=str,
                            choices=['Endgame', 'Normal'],
                            default='Endgame',
                            help="Select which MCTS to use, endgame visits each leaf node once "
                            )
        parser.add_argument('--max_elements_in_best_list', type=int,
                            default=10,
                            help='How many of the best results should be saved?')

        parser.add_argument('--prior_source', type=str,
                            default='neural_net',
                            choices=['neural_net', 'grammar', 'uniform'],
                            help='Select which source the prior used in MCTS should come from. '
                                 'neural_net uses the recognition object, '
                                 'grammar uses the probabilities from the grammar '
                                 'unifrom uses a uniform distribution above all options'
                            )
        parser.add_argument('--use-puct', type=str2bool,
                            default=True,
                            required=True,
                            help='Uses the PUCT formula when true, UCB1 otherwise')

        parser.add_argument('--temp_0', type=np.float32,
                            default=0.1,
                            help="Start temperature")

        parser.add_argument('--temperature_decay', type=np.float32,
                            default=-0.01,
                            help="Temperature for noise on "
                                                "actor prediction and MCTS "
                                                "prediction")
        parser.add_argument("--num_MCTS_sims", type=int,
                            required=True,
                            help="(int) Number of planning moves for MCTS to simulate")
        parser.add_argument("--c1", type=np.float32,
                            default=1.25,
                            help="(double) First exploration constant for MuZero in the PUCT formula")
        parser.add_argument("--gamma", type=np.float32,
                            default=0.98,
                            help="(double: [0, 1]) MDP Discounting factor for future rewards")
        parser.add_argument('--n_steps',  type=np.float32,
                            default=100,
                            help="(int > 0) Amount of steps to look ahead for rewards before bootstrapping (value function estimation)")
        parser.add_argument("--risk_seeking", type=str2bool,
                            default=True,
                            help='if Q-values in MCTS should be the maximum of its child Q-value.')
        parser.add_argument("--depth_first_search", type=str2bool,
                            default=True,
                            help='Weather the MCTS should do a complete roll out or not ')

        ## Replay buffer
        parser.add_argument("--replay_buffer_path", type=str,
                            default='', help="Path to a replay_buffer which should be loaded")

        parser.add_argument("--prioritize", type=str2bool,
                            default=False,
                            help="(bool) Set to true when using prioritized sampling from the replay buffer (used in Atari)")
        parser.add_argument("--prioritize_alpha", type=np.float32,
                            default=0.5,
                            help="(double) Exponentiation factor for computing probabilities in prioritized replay"),
        parser.add_argument("--prioritize_beta", type=np.float32,
                            default=1,
                            help="(double) Exponentiation factor for exponentiating the importance sampling ratio in prioritized replay")
        parser.add_argument('--selfplay_buffer_window', type=int,
                            default=50,
                            help="(int) Maximum number of self play iterations kept in the deque.")
        parser.add_argument("--balance_buffer", type=str2bool,
                            default=False,
                            help='Whether positive an negative samples in the buffer in the should be balanced  ')
        parser.add_argument("--max_percent_of_minimal_reward_runs_in_buffer", type=np.float32,
                            default=0.3,
                            help='How many percent of the examples in the buffer should maximal have a minimal reward')

        args = parser.parse_args()
        if args.seed is None:
            args.seed = int(time.time())
        return args
