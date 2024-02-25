from src.neural_nets.equation_encoder.equation_encoder_dummy import  EquationEncoderDummy
from src.neural_nets.equation_encoder.transformer_encoder_string import TransformerEncoderString

from src.neural_nets.data_set_encoder.measurement_encoder_dummy import MeasurementEncoderDummy
from src.neural_nets.data_set_encoder.mlp_encoder import MlpMeasurementEncoder
from src.neural_nets.data_set_encoder.measurement_encoder_picture import MeasurementEncoderPicture
from src.neural_nets.data_set_encoder.LSTM_measurement_encoder import LstmEncoder
from src.neural_nets.data_set_encoder.bi_LSTM_measurement_encoder import BiLSTMEncoder
from src.neural_nets.data_set_encoder.dataset_transformer import DatasetTransformer
from src.neural_nets.data_set_encoder.text_transformer import TextTransformer

from src.neural_nets.decoder.mlp_decoder import MLP_Decoder

from src.neural_nets.rule_predictor_net import  RulePredictorNet
def get_rule_predictor(args, reader_data):
    # EquationEncoder
    if args.class_equation_encoder == 'EquationEncoderDummy':
        encoder_tree_class = EquationEncoderDummy
    elif args.class_equation_encoder == 'Transformer_Encoder_String':
        encoder_tree_class = TransformerEncoderString
    else:
        raise ValueError(f'The class {args.class_equation_encoder} is not a valid class. '
                         'Please see --help to check options')

    # MeasurementEncoder
    if args.class_measurement_encoder == 'MeasurementEncoderDummy':
        encoder_measurement_class = MeasurementEncoderDummy
    elif args.class_measurement_encoder == 'LSTM_Measurement_Encoder':
        encoder_measurement_class = LstmEncoder
    elif args.class_measurement_encoder == 'Bi_LSTM_Measurement_Encoder':
        encoder_measurement_class = BiLSTMEncoder
    elif args.class_measurement_encoder == 'MLP_Measurement_Encoder':
        encoder_measurement_class = MlpMeasurementEncoder
    elif args.class_measurement_encoder == 'DatasetTransformer':
        encoder_measurement_class = DatasetTransformer
    elif args.class_measurement_encoder == "MeasurementEncoderPicture":
        encoder_measurement_class = MeasurementEncoderPicture
    elif args.class_measurement_encoder == "TextTransformer":
        encoder_measurement_class = TextTransformer

    else:
        raise ValueError(f'The class {args.class_measurement_encoder} is not a valid class. '
                         'Please see --help to check options')

    # Decoder for actor
    if args.actor_decoder_class == 'mlp_decoder':
        actor_decoder_class = MLP_Decoder
    else:
        raise ValueError(f'The class {args.actor_decoder} is not a valid class. '
                         'Please see --help to check options')

    # Decoder for actor
    if args.critic_decoder_class == 'mlp_decoder':
        critic_decoder_class = MLP_Decoder
    else:
        raise ValueError(f'The class {args.critic_decoder} is not a valid class. '
                         'Please see --help to check options')

    vocab_size = int(max(reader_data.vocab_size,
                     args.max_branching_factor ** (args.max_depth_of_tree + 2 )))

    input_feature_dims = get_input_feature_dim_for_DatasetTransformer(args, reader_data)

    rule_predictor = RulePredictorNet(
        encoder_tree_class=encoder_tree_class,
        encoder_tree_args=
        {
            'num_layers': args.num_layer_encoder_equation_transformer,
            'embedding_dim_encoder_equation': args.embedding_dim_encoder_equation,
            'num_heads': args.num_heads_encoder_equation_transformer,
            'dff': args.dim_feed_forward_equation_encoder_transformer,
            'vocab_size': vocab_size,
            'dropout_rate': args.dropout_rate,
            'encoder_measurements_LSTM_units': args.encoder_measurements_LSTM_units,
        },
        encoder_measurement_class=encoder_measurement_class,
        encoder_measurement_args=
        {
            'normalize_approach': args.normalize_approach,
            # LSTMS
            'encoder_measurements_LSTM_units': args.encoder_measurements_LSTM_units,
            'encoder_measurements_LSTM_return_sequence': args.encoder_measurements_LSTM_return_sequence,
            'encoder_measurement_num_layer' : args.encoder_measurement_num_layer,
            'encoder_measurement_num_neurons': args.encoder_measurement_num_neurons,
            'dropout_rate': args.dropout_rate,
            #------ DatasetTransformer
            # input_feature_dims states for every entry in the data how big is the encoding.
            'input_feature_dims': input_feature_dims,
            'stacking_depth': args.model_stacking_depth_dataset_transformer,
            'num_heads': args.model_num_heads_dataset_transformer,
            'model_dim_hidden': args.model_dim_hidden_dataset_transformer,
            'model_sep_res_embed': args.model_sep_res_embed_dataset_transformer,
            'model_att_block_layer_norm': args.model_att_block_layer_norm_dataset_transformer,
            'model_layer_norm_eps': args.model_layer_norm_eps_dataset_transformer,
            'model_att_score_norm': args.model_att_score_norm_dataset_transformer,
            'model_pre_layer_norm': args.model_pre_layer_norm_dataset_transformer,
            'model_rff_depth': args.model_rff_depth_dataset_transformer,

            'model_hidden_dropout_prob': args.model_hidden_dropout_prob_dataset_transformer,
            'model_att_score_dropout_prob': args.model_att_score_dropout_prob_dataset_transformer,
            'model_mix_heads': args.model_mix_heads_dataset_transformer,
            'model_embedding_layer_norm': args.model_embedding_layer_norm_dataset_transformer,
            'use_latent_vector': args.dataset_transformer_use_latent_vector,
            'bit_embedding': args.bit_embedding_dataset_transformer,
            'use_feature_index_embedding': args.use_feature_index_embedding_dataset_transformer,
            'max_len_datasets': args.max_len_datasets
            #ToDO Daniel add arguments you need

        },
        actor_decoder_class=actor_decoder_class,
        actor_decoder_args={
            'out_dim': reader_data.num_production_rules,
            'batch_sz': 1,
            'normalize_way': args.actor_decoder_normalize_way,
            'name': 'actor'
        },
        critic_decoder_class=critic_decoder_class,
        critic_decoder_args={
            'out_dim': 1,
            'batch_sz': 1,
            'normalize_way': args.critic_decoder_normalize_way,
            'name': 'critic'
        },
        args= args
    )
    return rule_predictor


def get_input_feature_dim_for_DatasetTransformer(args, reader_data):
    input_feature_dims = []
    for i in range(len(reader_data.dataset_columns) + 2 if "lin_transform" in args.normalize_approach else len(reader_data.dataset_columns) ):
        input_feature_dims.append( 32 if args.bit_embedding_dataset_transformer else 1)
    return input_feature_dims


def get_measurement_keys(column_names):
    measurement_columns = [name for name in column_names if 'row' in name]
    def custom_key(item):
        parts = item.split('_row_')
        if parts[0] == 'y':
            return(parts[1], 100000)
        else:
            i = parts[0].split('_')[1]
            return(parts[1], int(i))


    measurement_columns = sorted(measurement_columns, key=custom_key)
    return measurement_columns


def how_many_measurement_in_row(column_names):
    measurement_row_0_columns = [name for name in column_names if 'row_0' in name]
    return len(measurement_row_0_columns)

def get_column_names(column_names):
    measurement_row_0_columns = [name for name in column_names if 'row_0' in name]
    dataset_columns = [column.replace('_row_0', '') for column in measurement_row_0_columns]
    return dataset_columns


