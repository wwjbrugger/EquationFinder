## Load scripts and data to mogon
rsync -a  --verbose  \
/home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery \
mogon:/home/bruggerj \
--exclude=.idea --exclude=venv --exclude=.git \
--exclude=__pycache__ --exclude=results --exclude=wandb  \
--exclude=delete_me.txt  --exclude=usefull_ssh_comments.txt \
--exclude=data --exclude=out  --exclude=saved_models \
--exclude=.wandb --exclude=old --exclude=create_hpc_scripts.py\
--exclude=upload_wandb.py --exclude=data_grammar_2 --exclude=out \
--exclude=data_grammar_1  --exclude=data_grammar_4 \
--exclude=data_grammar_5 --exclude=data_grammar_6 \
--exclude=data_grammar_7 \
--exclude=old_wandb --exclude=.wandb_test \
--exclude=old_saved_models --exclude=old_data

## Load sbatch scripts to Mogon
rsync -a --verbose  \
/home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/scripts_final_nn \
mogon:/home/bruggerj/NeuralGuidedEquationDiscovery/ \

## Load output folder to computer
rsync -a --verbose  \
mogon:/home/bruggerj/NeuralGuidedEquationDiscovery/scripts_only_dataset_encoder/output/ \
/home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/scripts_only_dataset_encoder/output 


## Load saved models from mogon to computer 
rsync -a  -L --verbose    mogon:/home/bruggerj/NeuralGuidedEquationDiscovery/\
saved_models/  /home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/saved_models

## see jobs added to slurm manager 
 squeue -u  $USER --format="%.18i %.9P %.56j %.8T %.10M %.9l %.6D %R"

##  copy wandb files from mogon to computer
rsync -a  -L --verbose  mogon:/home/bruggerj/NeuralGuidedEquationDiscovery/\
.wandb/  /home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/.wandb



## initialite 
wandb sync -e jgu-wandb -p neural_guided_symbolic_regression
                      f" --id {id}" \
                      f" {path_to_run_to_upload}"
