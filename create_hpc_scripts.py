import itertools
from pathlib import Path


def run():
    parameter_list_dict = {
        # paprameter to change
        'experiment_name': ['train_pretrained_net'],
        'minutes_to_run': ['600'],
        'max_iteration_to_run': [200],
        'seed': ['$SLURM_ARRAY_TASK_ID'],
        'path_to_complete_model': [''],
        'path_to_pretrained_dataset_encoder': [
           'saved_models/run_1/test_dataset_encoder__neural_net__data_grammar_1_run_1__DatasetTransformer__abs_max_y__lin_transform/5/tf_ckpts/ckpt-399'
    ],
        'replay_buffer_path': [''],
        'run_mcts': [True],
        'only_test': [False],
        'data': [
            'data_grammar_1/run_1',
            # 'data_grammar_1/nguyen_1',
            # 'data_grammar_1/nguyen_2',
            # 'data_grammar_1/nguyen_3',
            # 'data_grammar_1/nguyen_4',
            # 'data_grammar_1/nguyen_5',
            # 'data_grammar_1/nguyen_6',
            # 'data_grammar_1/nguyen_7',
            # 'data_grammar_1/nguyen_8',
            # 'data_grammar_1/nguyen_9',
            # 'data_grammar_1/nguyen_10',
            # 'data_grammar_1/nguyen_11',
            # 'data_grammar_1/nguyen_12',
            # 'data_grammar_2/nguyen_1',
            # 'data_grammar_2/nguyen_2',
            # 'data_grammar_2/nguyen_3',
            # 'data_grammar_2/nguyen_4',
            # 'data_grammar_2/nguyen_5',
            # 'data_grammar_2/nguyen_6',
            # 'data_grammar_2/nguyen_7',
            # 'data_grammar_2/nguyen_8',
            # 'data_grammar_2/nguyen_9',
            # 'data_grammar_2/nguyen_10',
            # 'data_grammar_2/nguyen_11',
            # 'data_grammar_2/nguyen_12'
        ],

        'script_folder': ['scripts_final'],
        'output_folder': ['output'],
        'cold_start_iterations': [20],
        'class_measurement_encoder': [
            # 'MeasurementEncoderDummy',
            # 'LSTM_Measurement_Encoder',
            # 'Bi_LSTM_Measurement_Encoder',
            # 'MLP_Measurement_Encoder',
              'DatasetTransformer',
            # 'MeasurementEncoderPicture',
            # 'TextTransformer'
        ],
        'prior_source': ['neural_net'],  # 'neural_net''grammar', 'uniform'
        'normalize_approach': [
            # 'None'
            # 'abs_max_y',
            # 'row_wise',
             'abs_max_y__lin_transform',
            # 'abs_max_y',
        ],
        'num_MCTS_sims': [-1],
        'sec_per_simulation': [30],
        'MCTS_engine': ['Endgame'],
        ## General

        'logging_level': ['30'],
        'wandb': ['offline'],
        'gpu': ['0'],

        'num_selfplay_iterations': ['10'],
        'num_selfplay_iterations_test': ['1'],
        'test_network': ['True'],
        'test_every_n_steps': [1],
        ## Infos about Tree
        'minimum_reward': ['-1'],
        'maximum_reward': ['1'],
        'max_depth_of_tree': ['10'],
        'max_num_nodes_in_syntax_tree': ['30'],
        'max_branching_factor': [2],
        'max_constants_in_tree': [3],
        ## Training neural net
        'batch_size_training': ['16'],
        'num_gradient_steps': ['200'],
        'average_policy_if_wrong': [False],

        ## Preprocess
        'equation_preprocess_class': [
            'PandasPreprocess'
        ],
        'max_len_datasets': [20],
        ## Encoder Equations
        'class_equation_encoder': ['Transformer_Encoder_String'],
        'embedding_dim_encoder_equation': ['8'],
        'max_tokens_equation': ['64'],
        'use_position_encoding': ['False'],
        ## Attention String
        'num_layer_encoder_equation_transformer': ['2'],
        'num_heads_encoder_equation_transformer': ['4'],
        'dim_feed_forward_equation_encoder_transformer': ['32'],
        'dropout_rate': ['0.1'],
        ## Encoder Measurement

        'contrastive_loss': [True],
        ## Encoder Measurement LSTM
        'encoder_measurements_LSTM_units': ['64'],
        'encoder_measurements_LSTM_return_sequence': ['True'],
        ## Encoder Measurement  MLP
        'encoder_measurement_num_layer': [3],
        'encoder_measurement_num_neurons': [64],

        ######### Encoder Measurement Transformer
        'model_dim_hidden_dataset_transformer': [64],
        'model_num_heads_dataset_transformer': [8],
        'model_stacking_depth_dataset_transformer': [4],
        'model_sep_res_embed_dataset_transformer': ['True'],
        'model_att_block_layer_norm_dataset_transformer': ['True'],
        'model_layer_norm_eps_dataset_transformer': [1e-12],
        'model_att_score_norm_dataset_transformer': ['softmax'],
        'model_pre_layer_norm_dataset_transformer': ['False'],
        'model_rff_depth_dataset_transformer': [2],
        'model_hidden_dropout_prob_dataset_transformer': [1e-06],
        'model_att_score_dropout_prob_dataset_transformer': [1e-06],
        'model_mix_heads_dataset_transformer': ['True'],
        'model_embedding_layer_norm_dataset_transformer': ['False'],
        'bit_embedding_dataset_transformer': ['False'],
        'use_feature_index_embedding_dataset_transformer': [False],
        'dataset_transformer_use_latent_vector': ['True'],
        ######### Encoder Measurement TextTransformer
        'float_precision_text_transformer':[3],
        'mantissa_len_text_transformer': [1],
        'max_exponent_text_transformer': [100],
        'num_dimensions_text_transformer': [3],
        'embedding_dim_text_transformer': [512],
        'embedder_intermediate_expansion_factor_text_transformer': [1.0],
        'num_encoder_layers_text_transformer': [4],
        'num_attention_heads_text_transformer': [8],
        'encoder_intermediate_expansion_factor_text_transformer': [4.0],
        'intermediate_dropout_rate_text_transformer': [0.2],
        'attention_dropout_rate_text_transformer': [0.1],
        ## Actor Decoder
        'actor_decoder_class': ['mlp_decoder'],
        'actor_decoder_normalize_way': ['soft_max'],
        ## Critic Decoder
        'critic_decoder_class': ['mlp_decoder'],
        'critic_decoder_normalize_way': ['tanh'],
        ## MCTS

        'max_elements_in_list': [10],
        'use_puct': [True],
        'temp_0': [0.1],
        'temperature_decay': ['-0.01'],
        'c1': ['1.25'],
        'gamma': ['0.98'],
        'n_steps': ['100'],
        'risk_seeking': [True],
        'depth_first_search': [True],
        ## Replay buffer

        'prioritize': ['False'],
        'prioritize_alpha': ['0.5'],
        'prioritize_beta': ['1'],
        'selfplay_buffer_window': ['50'],
        'balance_buffer': [False],
        'max_percent_of_minimal_reward_runs_in_buffer': [0.3]
    }
    create_output_folder(parameter_list_dict)

    cartesian_product = itertools.product(
        *parameter_list_dict.values(),
        repeat=1
    )
    created_experiments = []
    for values in cartesian_product:
        settings_one_script = dict(zip(parameter_list_dict.keys(), values))
        settings_one_script['experiment_name'] = create_experiment_name(
            settings_one_script)
        write_script(settings_one_script)
        created_experiments.append(settings_one_script['experiment_name'])

    write_experiment_names_to_file(
        experiment_list=created_experiments,
        script_folder=parameter_list_dict['script_folder'][0]
    )
    write_sbatch_comments_to_file(
        experiment_list=created_experiments,
        script_folder=parameter_list_dict['script_folder'][0]
    )


def create_experiment_name(settings_one_script):
    basic_experiment_name = settings_one_script['experiment_name']
    experiment_name = f"{basic_experiment_name}__" \
                      f"{settings_one_script['prior_source']}__" \
                      f"{settings_one_script['data'].replace('/','_')}__" \
                      f"{settings_one_script['class_measurement_encoder']}__" \
                      f"{settings_one_script['normalize_approach']}__" \
                      f"{settings_one_script['MCTS_engine']}__" \
                      f"{settings_one_script['sec_per_simulation']}"

    return experiment_name


def create_output_folder(parameter_list_dict):
    path = Path(
        f"{parameter_list_dict['script_folder'][0]}/"
        f"{parameter_list_dict['output_folder'][0]}")
    path.mkdir(exist_ok=True, parents=True)
    with open(path / 'delete_me.txt', "w") as file:
        file.writelines(
            "This file exists for Mogon to create the output folder. ")


def write_script(settings_one_script):
    with open(f"{settings_one_script['script_folder']}/"
              f"{settings_one_script['experiment_name']}.sh", "w") as file1:
        write_SBATCH_commants(settings_one_script, file1)
        write_prepare_enviroment(file1)
        write_python_call(settings_one_script, file1)


def write_SBATCH_commants(settings_one_script, file1):
    file1.write(f"#!/bin/bash\n")
    file1.write(
        f"#SBATCH --job-name={settings_one_script['experiment_name']} \n")
    file1.writelines("#SBATCH -p smp \n")
    file1.writelines("#SBATCH --account=m2_datamining \n")
    file1.writelines(
        f"#SBATCH --time={int(settings_one_script['minutes_to_run']) + 100} \n")
    file1.writelines("#SBATCH --tasks=1 \n")
    file1.writelines("#SBATCH --nodes=1 \n")
    file1.writelines("#SBATCH --cpus-per-task=4 \n")
    file1.writelines("#SBATCH --mem=20GB \n")
    file1.writelines("#SBATCH --array=1-1 \n")
    file1.writelines("\n")
    file1.writelines("#SBATCH -o \%x_\%j_profile.out \n")
    file1.writelines("#SBATCH -C anyarch \n")
    file1.writelines(f"#SBATCH -o {settings_one_script['output_folder']}"
                     f"/\%x_\%j.out \n")
    file1.writelines(f"#SBATCH -e {settings_one_script['output_folder']}"
                     f"/\%x_\%j.err \n")
    file1.writelines("#SBATCH --mail-user=bruggerj@uni-mainz.de \n")
    file1.writelines("#SBATCH --mail-type=FAIL \n")
    file1.writelines("\n")


def write_prepare_enviroment(file1):
    file1.writelines("module purge \n")
    file1.writelines("module load "
                     " math/CPLEX/22.10-GCCcore-11.2.0-Python-3.9.6\n"
                     )
    file1.writelines("cd ..\n")
    file1.writelines("export PYTHONPATH=$PYTHONPATH:$(pwd) \n")
    file1.writelines(
        "export http_proxy=http://webproxy.zdv.uni-mainz.de:8888 \n")
    file1.writelines(
        "export https_proxy=https://webproxy.zdv.uni-mainz.de:8888 \n")
    file1.writelines("source venv/bin/activate\n")
    file1.writelines("wandb offline \n")
    file1.writelines("\n")
    file1.writelines("\n")


def write_python_call(settings_one_script, file1):
    file1.writelines(f"srun python "
                     f"src/start_nged.py \\\n")
    ## General
    if len(settings_one_script['path_to_complete_model']) > 5:
        file1.writelines(f"--path_to_complete_model {settings_one_script['path_to_complete_model']} \\\n")
    if len(settings_one_script['path_to_pretrained_dataset_encoder']) > 5:
        file1.writelines(f"--path_to_pretrained_dataset_encoder {settings_one_script['path_to_pretrained_dataset_encoder']} \\\n")

    file1.writelines(f"--experiment_name $SLURM_JOB_NAME \\\n")
    file1.writelines(f"--job_id $SLURM_JOB_ID \\\n")
    file1.writelines(f"--minutes_to_run {settings_one_script['minutes_to_run']} \\\n")
    file1.writelines(f"--max_iteration_to_run {settings_one_script['max_iteration_to_run']} \\\n")
    file1.writelines(f"--sec_per_simulation {settings_one_script['sec_per_simulation']} \\\n")

    file1.writelines(
        f"--max_num_nodes_in_syntax_tree {settings_one_script['max_num_nodes_in_syntax_tree']} \\\n")
    file1.writelines(f"--run_mcts {settings_one_script['run_mcts']} \\\n")
    file1.writelines(f"--only_test {settings_one_script['only_test']} \\\n")
    file1.writelines(f"--seed {settings_one_script['seed']} \\\n")
    file1.writelines(
        f"--logging_level {settings_one_script['logging_level']} \\\n")
    file1.writelines(f"--wandb {settings_one_script['wandb']} \\\n")
    file1.writelines(f"--gpu {settings_one_script['gpu']} \\\n")
    file1.writelines(f"--data {settings_one_script['data']} \\\n")
    file1.writelines(
        f"--num_selfplay_iterations {settings_one_script['num_selfplay_iterations']} \\\n")
    file1.writelines(
        f"--num_selfplay_iterations_test {settings_one_script['num_selfplay_iterations_test']} \\\n")
    file1.writelines(
        f"--test_network {settings_one_script['test_network']} \\\n")
    file1.writelines(
        f"--test_every_n_steps {settings_one_script['test_every_n_steps']} \\\n"
    )
    ## Infos about Tree
    file1.writelines(
        f"--minimum_reward {settings_one_script['minimum_reward']} \\\n")
    file1.writelines(
        f"--maximum_reward {settings_one_script['maximum_reward']} \\\n")
    file1.writelines(
        f"--max_depth_of_tree {settings_one_script['max_depth_of_tree']} \\\n")
    file1.writelines(
        f"--max_branching_factor {settings_one_script['max_branching_factor']} \\\n")
    file1.writelines(
        f"--max_constants_in_tree {settings_one_script['max_constants_in_tree']} \\\n")
    ## Training neural net
    file1.writelines(
        f"--batch_size_training {settings_one_script['batch_size_training']} \\\n")
    file1.writelines(
        f"--num_gradient_steps {settings_one_script['num_gradient_steps']} \\\n")
    file1.writelines(
        f"--average_policy_if_wrong {settings_one_script['average_policy_if_wrong']} \\\n")
    file1.writelines(
        f"--cold_start_iterations {settings_one_script['cold_start_iterations']} \\\n")
    ## Preprocess
    file1.writelines(
        f"--equation_preprocess_class {settings_one_script['equation_preprocess_class']} \\\n")
    file1.writelines(
        f"--max_len_datasets {settings_one_script['max_len_datasets']} \\\n")
    ## Encoder Equations
    file1.writelines(
        f"--class_equation_encoder {settings_one_script['class_equation_encoder']} \\\n")
    file1.writelines(
        f"--embedding_dim_encoder_equation {settings_one_script['embedding_dim_encoder_equation']} \\\n")
    file1.writelines(
        f"--max_tokens_equation {settings_one_script['max_tokens_equation']} \\\n")
    file1.writelines(
        f"--use_position_encoding {settings_one_script['use_position_encoding']} \\\n")
    ## Attention String
    file1.writelines(
        f"--num_layer_encoder_equation_transformer {settings_one_script['num_layer_encoder_equation_transformer']} \\\n")
    file1.writelines(
        f"--num_heads_encoder_equation_transformer {settings_one_script['num_heads_encoder_equation_transformer']} \\\n")
    file1.writelines(
        f"--dim_feed_forward_equation_encoder_transformer {settings_one_script['dim_feed_forward_equation_encoder_transformer']} \\\n")
    file1.writelines(
        f"--dropout_rate {settings_one_script['dropout_rate']} \\\n")
    ## Encoder Measurement
    file1.writelines(
        f"--class_measurement_encoder {settings_one_script['class_measurement_encoder']} \\\n")
    file1.writelines(
        f"--normalize_approach {settings_one_script['normalize_approach']} \\\n")
    file1.writelines(
        f"--contrastive_loss {settings_one_script['contrastive_loss']} \\\n")

    ## Encoder Measurement LSTM
    file1.writelines(
        f"--encoder_measurements_LSTM_units {settings_one_script['encoder_measurements_LSTM_units']} \\\n")
    file1.writelines(
        f"--encoder_measurements_LSTM_return_sequence {settings_one_script['encoder_measurements_LSTM_return_sequence']} \\\n")

    ## Encoder Measurement  MLP
    file1.writelines(
        f"--encoder_measurement_num_layer {settings_one_script['encoder_measurement_num_layer']} \\\n")
    file1.writelines(
        f"--encoder_measurement_num_neurons {settings_one_script['encoder_measurement_num_neurons']} \\\n")
    ######### Encoder Measurement Transformer
    file1.writelines(
        f"--model_dim_hidden_dataset_transformer {settings_one_script['model_dim_hidden_dataset_transformer']} \\\n")
    file1.writelines(
        f"--model_num_heads_dataset_transformer {settings_one_script['model_num_heads_dataset_transformer']} \\\n")
    file1.writelines(
        f"--model_stacking_depth_dataset_transformer {settings_one_script['model_stacking_depth_dataset_transformer']} \\\n")
    file1.writelines(
        f"--model_sep_res_embed_dataset_transformer {settings_one_script['model_sep_res_embed_dataset_transformer']} \\\n")
    file1.writelines(
        f"--model_att_block_layer_norm_dataset_transformer {settings_one_script['model_att_block_layer_norm_dataset_transformer']} \\\n")
    file1.writelines(
        f"--model_layer_norm_eps_dataset_transformer {settings_one_script['model_layer_norm_eps_dataset_transformer']} \\\n")
    file1.writelines(
        f"--model_att_score_norm_dataset_transformer {settings_one_script['model_att_score_norm_dataset_transformer']} \\\n")
    file1.writelines(
        f"--model_pre_layer_norm_dataset_transformer {settings_one_script['model_pre_layer_norm_dataset_transformer']} \\\n")
    file1.writelines(
        f"--model_rff_depth_dataset_transformer {settings_one_script['model_rff_depth_dataset_transformer']} \\\n")
    file1.writelines(
        f"--model_hidden_dropout_prob_dataset_transformer {settings_one_script['model_hidden_dropout_prob_dataset_transformer']} \\\n")
    file1.writelines(
        f"--model_att_score_dropout_prob_dataset_transformer {settings_one_script['model_att_score_dropout_prob_dataset_transformer']} \\\n")
    file1.writelines(
        f"--model_mix_heads_dataset_transformer {settings_one_script['model_mix_heads_dataset_transformer']} \\\n")
    file1.writelines(
        f"--model_embedding_layer_norm_dataset_transformer {settings_one_script['model_embedding_layer_norm_dataset_transformer']} \\\n")
    file1.writelines(
        f"--bit_embedding_dataset_transformer {settings_one_script['bit_embedding_dataset_transformer']} \\\n")
    file1.writelines(
        f"--dataset_transformer_use_latent_vector {settings_one_script['dataset_transformer_use_latent_vector']} \\\n")
    file1.writelines(
        f"--use_feature_index_embedding_dataset_transformer {settings_one_script['use_feature_index_embedding_dataset_transformer']} \\\n")
    ####### Encoder Measurement TextTransformer
    file1.writelines(
        f"--float_precision_text_transformer {settings_one_script['float_precision_text_transformer']} \\\n")
    file1.writelines(
        f"--mantissa_len_text_transformer {settings_one_script['mantissa_len_text_transformer']} \\\n")
    file1.writelines(
        f"--max_exponent_text_transformer {settings_one_script['max_exponent_text_transformer']} \\\n")
    file1.writelines(
        f"--num_dimensions_text_transformer {settings_one_script['num_dimensions_text_transformer']} \\\n")
    file1.writelines(
        f"--embedding_dim_text_transformer {settings_one_script['embedding_dim_text_transformer']} \\\n")
    file1.writelines(
        f"--embedder_intermediate_expansion_factor_text_transformer {settings_one_script['embedder_intermediate_expansion_factor_text_transformer']} \\\n")
    file1.writelines(
        f"--num_encoder_layers_text_transformer {settings_one_script['num_encoder_layers_text_transformer']} \\\n")
    file1.writelines(
        f"--num_attention_heads_text_transformer {settings_one_script['num_attention_heads_text_transformer']} \\\n")
    file1.writelines(
        f"--encoder_intermediate_expansion_factor_text_transformer {settings_one_script['encoder_intermediate_expansion_factor_text_transformer']} \\\n")
    file1.writelines(
        f"--intermediate_dropout_rate_text_transformer {settings_one_script['intermediate_dropout_rate_text_transformer']} \\\n")
    file1.writelines(
        f"--attention_dropout_rate_text_transformer {settings_one_script['attention_dropout_rate_text_transformer']} \\\n")

    ## Actor Decoder
    file1.writelines(
        f"--actor_decoder_class {settings_one_script['actor_decoder_class']} \\\n")
    file1.writelines(
        f"--actor_decoder_normalize_way {settings_one_script['actor_decoder_normalize_way']} \\\n")
    ## Critic Decoder 
    file1.writelines(
        f"--critic_decoder_class {settings_one_script['critic_decoder_class']} \\\n")
    file1.writelines(
        f"--critic_decoder_normalize_way {settings_one_script['critic_decoder_normalize_way']} \\\n")

    ## MCTS
    file1.writelines(
        f"--MCTS_engine {settings_one_script['MCTS_engine']} \\\n")
    file1.writelines(
        f"--max_elements_in_best_list {settings_one_script['max_elements_in_list']} \\\n")
    file1.writelines(
        f"--prior_source {settings_one_script['prior_source']} \\\n")
    file1.writelines(
        f"--temp_0 {settings_one_script['temp_0']} \\\n")
    file1.writelines(
        f"--temperature_decay {settings_one_script['temperature_decay']} \\\n")
    file1.writelines(
        f"--num_MCTS_sims {settings_one_script['num_MCTS_sims']} \\\n")
    file1.writelines(f"--c1 {settings_one_script['c1']} \\\n")
    file1.writelines(f"--gamma {settings_one_script['gamma']} \\\n")
    file1.writelines(f"--n_steps {settings_one_script['n_steps']} \\\n")
    file1.writelines(f"--risk_seeking {settings_one_script['risk_seeking']} \\\n")
    file1.writelines(f"--depth_first_search {settings_one_script['depth_first_search']} \\\n")
    ## Replay buffer
    if len(settings_one_script['replay_buffer_path']) > 5:
        file1.writelines(f"--replay_buffer_path {settings_one_script['path_to_complete_model']} \\\n")
    file1.writelines(f"--prioritize {settings_one_script['prioritize']} \\\n")
    file1.writelines(
        f"--prioritize_alpha {settings_one_script['prioritize_alpha']} \\\n")
    file1.writelines(
        f"--prioritize_beta {settings_one_script['prioritize_beta']} \\\n")
    file1.writelines(
        f"--selfplay_buffer_window {settings_one_script['selfplay_buffer_window']} \\\n")
    file1.writelines(
        f"--balance_buffer {settings_one_script['balance_buffer']} \\\n")
    file1.writelines(
        f"--max_percent_of_minimal_reward_runs_in_buffer"
        f" {settings_one_script['max_percent_of_minimal_reward_runs_in_buffer']} \\\n")
    file1.writelines(
        f"--use-puct"
        f" {settings_one_script['use_puct']} \\\n")


def write_experiment_names_to_file(experiment_list, script_folder):
    with open(f"{script_folder}/experiment_name.txt", "a") as file2:
        for experiment_name in experiment_list:
            file2.write(f"\"{experiment_name}\",\n")


def write_sbatch_comments_to_file(experiment_list, script_folder):
    with open(f"{script_folder}/sbatch_comments.txt", "a") as file2:
        for experiment_name in experiment_list:
            file2.write(f"sbatch {experiment_name}.sh \n")


if __name__ == '__main__':
    run()
