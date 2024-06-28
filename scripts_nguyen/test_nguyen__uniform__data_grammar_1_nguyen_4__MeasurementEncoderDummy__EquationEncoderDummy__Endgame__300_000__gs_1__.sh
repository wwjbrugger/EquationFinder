#!/bin/bash
#SBATCH --job-name=test_nguyen__uniform__data_grammar_1_nguyen_4__MeasurementEncoderDummy__EquationEncoderDummy__Endgame__300_000__gs_1__ 
#SBATCH -p smp 
#SBATCH --account=m2_datamining 
#SBATCH --time=310 
#SBATCH --tasks=1 
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4 
#SBATCH --mem=24GB 
#SBATCH --array=1-4

#SBATCH -o \%x_\%j_\%A_\%a_profile.out 
#SBATCH -C anyarch 
#SBATCH -o output/\%x_\%j_\%A_\%a.out 
#SBATCH -e output/\%x_\%j_\%A_\%a.err 
#SBATCH --mail-user=bruggerj@uni-mainz.de 
#SBATCH --mail-type=FAIL 

module purge 
module load  math/CPLEX/22.10-GCCcore-11.2.0-Python-3.9.6
cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd) 
export http_proxy=http://webproxy.zdv.uni-mainz.de:8888 
export https_proxy=https://webproxy.zdv.uni-mainz.de:8888 
source ~/NeuralGuidedEquationDiscovery/venv/bin/activate
wandb offline 


srun python src/start_nged.py \
--experiment_name $SLURM_JOB_NAME \
--grammar_search 1 \
--grammar_for_generation None \
--job_id $SLURM_JOB_ID \
--minutes_to_run 300 \
--max_iteration_to_run 1 \
--max_num_nodes_in_syntax_tree 25 \
--generate_new_training_data False \
--only_test True \
--seed $SLURM_ARRAY_TASK_ID \
--logging_level 20 \
--training_mode mcts \
--wandb offline \
--gpu 0 \
--data data_grammar_1/nguyen_4 \
--num_selfplay_iterations 10 \
--num_selfplay_iterations_test 1 \
--test_network True \
--test_every_n_steps 1 \
--max_ram 20 \
--max_run_time 300 \
--minimum_reward -1 \
--maximum_reward 1 \
--build_syntax_tree_token_based False \
--max_depth_of_tree 10 \
--max_branching_factor 2 \
--max_constants_in_tree 2 \
--batch_size_training 64 \
--num_gradient_steps 100 \
--average_policy_if_wrong True \
--cold_start_iterations 10 \
--equation_preprocess_class PandasPreprocess \
--max_len_datasets 20 \
--class_equation_encoder EquationEncoderDummy \
--embedding_dim_encoder_equation 8 \
--max_tokens_equation 64 \
--use_position_encoding False \
--num_layer_encoder_equation_transformer 2 \
--num_heads_encoder_equation_transformer 4 \
--dim_feed_forward_equation_encoder_transformer 32 \
--dropout_rate 0.1 \
--class_measurement_encoder MeasurementEncoderDummy \
--normalize_approach abs_max_y \
--contrastive_loss False \
--encoder_measurements_LSTM_units 64 \
--encoder_measurements_LSTM_return_sequence True \
--encoder_measurement_num_layer 3 \
--encoder_measurement_num_neurons 128 \
--model_dim_hidden_dataset_transformer 128 \
--model_num_heads_dataset_transformer 8 \
--model_stacking_depth_dataset_transformer 4 \
--model_sep_res_embed_dataset_transformer True \
--model_att_block_layer_norm_dataset_transformer True \
--model_layer_norm_eps_dataset_transformer 1e-12 \
--model_att_score_norm_dataset_transformer softmax \
--model_pre_layer_norm_dataset_transformer False \
--model_rff_depth_dataset_transformer 2 \
--model_hidden_dropout_prob_dataset_transformer 1e-06 \
--model_att_score_dropout_prob_dataset_transformer 1e-06 \
--model_mix_heads_dataset_transformer True \
--model_embedding_layer_norm_dataset_transformer False \
--bit_embedding_dataset_transformer False \
--dataset_transformer_use_latent_vector True \
--use_feature_index_embedding_dataset_transformer False \
--float_precision_text_transformer 3 \
--mantissa_len_text_transformer 1 \
--max_exponent_text_transformer 100 \
--num_dimensions_text_transformer 3 \
--embedding_dim_text_transformer 512 \
--embedder_intermediate_expansion_factor_text_transformer 1.0 \
--num_encoder_layers_text_transformer 4 \
--num_attention_heads_text_transformer 8 \
--encoder_intermediate_expansion_factor_text_transformer 4.0 \
--intermediate_dropout_rate_text_transformer 0.2 \
--attention_dropout_rate_text_transformer 0.1 \
--actor_decoder_class mlp_decoder \
--actor_decoder_normalize_way soft_max \
--critic_decoder_class mlp_decoder \
--critic_decoder_normalize_way tanh \
--MCTS_engine Endgame \
--max_elements_in_best_list 10 \
--prior_source uniform \
--temp_0 0.5 \
--temperature_decay 0 \
--num_MCTS_sims 300_000 \
--c1 10 \
--gamma 1 \
--n_steps 100 \
--risk_seeking True \
--depth_first_search True \
--prioritize False \
--prioritize_alpha 0.5 \
--prioritize_beta 1 \
--selfplay_buffer_window 50 \
--balance_buffer False \
--max_percent_of_minimal_reward_runs_in_buffer 0.3 \
--use-puct True \
--old_run False \
