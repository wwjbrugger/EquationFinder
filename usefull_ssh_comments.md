## Load scripts and data to mogon
rsync -a  --verbose  \
/home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/ \
mogon:/home/bruggerj/NeuralGuidedEquationDiscoveryTest \
--exclude=.idea --exclude=venv --exclude=.git \
--exclude=__pycache__ --exclude=results --exclude=wandb  \
--exclude=delete_me.txt  --exclude=usefull_ssh_comments.txt \
--exclude=data --exclude=out  --exclude=old_saved_models \
--exclude=.wandb --exclude=old_data --exclude=create_hpc_scripts.py\
--exclude=upload_wandb.py --exclude=out  \
--exclude=data_grammar_1  --exclude=data_grammar_2 --exclude=data_grammar_3 \
--exclude=data_grammar_6  --exclude=data_grammar_5 --exclude=data_grammar_4 \
--exclude=data_grammar_7 --exclude=data_token_based \
--exclude=saved_models/data_grammar_4 --exclude=saved_models/data_grammar_5 \
--exclude=saved_models/data_grammar_6 --exclude=saved_models/data_grammar_7 \
--exclude=data_grammar_2/run_1 --exclude=old_wandb --exclude=.wandb_test  \
--exclude=saved_models --exclude=saved_models/run_3/only* \
--exclude=scripts_token --exclude=scripts_train_complete_model --exclude=scripts_only_dataset_encoder

## saved models to mogon
rsync -a  --verbose  \
/home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/ \
mogon:/home/bruggerj/NeuralGuidedEquationDiscoveryTest \

## Load saved models from computer to mogon

rsync -avz -L --include='*/' --include='*ckpt-9*' --exclude='*' \
-e ssh /home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/saved_models/data_grammar_8 \
mogon:/home/bruggerj/NeuralGuidedEquationDiscoveryTest/saved_models


## Load sbatch scripts to Mogon
rsync -a --verbose  \
/home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/scripts_final_nn \
mogon:/home/bruggerj/NeuralGuidedEquationDiscovery/ \

## Load saved models from mogon to computer 
rsync -a  -L --verbose    mogon:/home/bruggerj/NeuralGuidedEquationDiscovery/\
saved_models/  /home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/saved_models



## see jobs added to slurm manager 
 squeue -u  $USER --format="%.18i %.9P %.56j %.8T %.10M %.9l %.6D %R"

##  copy wandb files from mogon to computer
rsync  -a  -L --verbose  mogon:/home/bruggerj/NeuralGuidedEquationDiscoveryTest/\
.wandb/  /home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/.wandb_test \




## initialite 
wandb sync -e jgu-wandb -p neural_guided_symbolic_regression
                      f" --id {id}" \
                      f" {path_to_run_to_upload}"
