#!/bin/bash
#-----------------------------------------------------------------
# Example SLURM job script to run serial applications on Mogon.
#
# This script requests one task using 2 cores on one GPU-node.
#-----------------------------------------------------------------

#SBATCH -J Bi_LSTM_Measurement_Encoder_lin_Endgame_grammar_1_nguyen_9        # Job name
#SBATCH -o \%x_\%j_profile.out
#SBATCH -p smp                # Partition name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=120
#SBATCH -C anyarch
#SBATCH --account=m2_datamining             # Specify allocation to charge against
#SBATCH -o output_mogon/\%x_\%j.out
#SBATCH -e output_mogon/\%x_\%j.err
#SBATCH --mail-user=bruggerj@uni-mainz.de
#SBATCH --mail-type=END

# Load all necessary modules if needed (these are examples)
# Loading modules in the script ensures a consistent environment.
module purge
module load  math/CPLEX/22.10-GCCcore-11.2.0-Python-3.9.6
cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd)
export http_proxy=http://webproxy.zdv.uni-mainz.de:8888
export https_proxy=http://webproxy.zdv.uni-mainz.de:8888
source venv/bin/activate
wandb offline
# Launch the executable
srun python src/start_nged.py --path_to_complete_model  saved_models/run_1/AlphaZero/Bi_LSTM_Measurement_Encoder_lin/1/tf_ckpts/ckpt-184 --experiment_name Bi_LSTM_Measurement_Encoder_lin_Endgame_grammar_1_nguyen_9 --max_num_nodes_in_syntax_tree 30 --replay_buffer_path None --run_mcts False --only_test True --seed 0 --minutes_to_run 1 --logging_level 20 --wandb offline --gpu 0 --data data_grammar_1/nguyen_9 --num_selfplay_iterations 10 --num_selfplay_iterations_test 20 --test_network True --test_every_n_steps 1 --minimum_reward -1 --maximum_reward 1 --max_depth_of_tree 10 --max_branching_factor 2 --max_constants_in_tree 2 --batch_size_training 16 --num_gradient_steps 20 --average_policy_if_wrong False --cold_start_iterations 0 --equation_preprocess_class PandasPreprocess --max_len_datasets 20 --class_equation_encoder Transformer_Encoder_String --embedding_dim_encoder_equation 8 --max_tokens_equation 64 --use_position_encoding False --num_layer_encoder_equation_transformer 2 --num_heads_encoder_equation_transformer 4 --dim_feed_forward_equation_encoder_transformer 32 --dropout_rate 0.1 --class_measurement_encoder Bi_LSTM_Measurement_Encoder --normalize_approach abs_max_y__lin_transform --contrastive_loss True --encoder_measurements_LSTM_units 64 --encoder_measurements_LSTM_return_sequence True --encoder_measurement_num_layer 3 --encoder_measurement_num_neurons 64 --model_dim_hidden_dataset_transformer 64 --model_num_heads_dataset_transformer 8 --model_stacking_depth_dataset_transformer 4 --model_sep_res_embed_dataset_transformer True --model_att_block_layer_norm_dataset_transformer True --model_layer_norm_eps_dataset_transformer 1e-12 --model_att_score_norm_dataset_transformer softmax --model_pre_layer_norm_dataset_transformer False --model_rff_depth_dataset_transformer 2 --model_hidden_dropout_prob_dataset_transformer 1e-06 --model_att_score_dropout_prob_dataset_transformer 1e-06 --model_mix_heads_dataset_transformer True --model_embedding_layer_norm_dataset_transformer False --bit_embedding_dataset_transformer False --dataset_transformer_use_latent_vector True --actor_decoder_class mlp_decoder --actor_decoder_normalize_way positive_sum_1 --critic_decoder_class mlp_decoder --critic_decoder_normalize_way tanh --MCTS_engine Endgame --max_elements_in_best_list 10 --prior_source neural_net --temp_0 0.1 --temperature_decay -0.01 --num_MCTS_sims 10000000 --c1 1.25 --gamma 0.98 --n_steps 100 --risk_seeking True --depth_first_search True --prioritize False --prioritize_alpha 0.5 --prioritize_beta 1 --selfplay_buffer_window 50 --balance_buffer False --max_percent_of_minimal_reward_runs_in_buffer 0.3 --use-puct False
