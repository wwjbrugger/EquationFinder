# Welcome to the accompanying repro of the paper Neural-Guided Equation Discovery

MGMD is a modular equation discovery system with neural-guided MCTS that can be trained using supervised or reinforcement learning.

The entrance point of the project is src/start_nged.py.
Check src/config_ngedn.py to see possible configurations and their explanation. 


# How can I reproduce the result from the paper? 

The Experiments are performed on Linux with Python 3.9 

To install a minimal set of packages, use 'pip install -r requirements_pipreqs.txt'. 
All installed packages can be installed with 'pip install -r requirements_pip.txt'

For training and measuring the contrastive loss of the tabular data set embedding architectures use the main branch and the scripts in 
'scripts_train_complete_model' and  'scripts_only_dataset_encoder'. 

For testing the trained architectures, use the branch 'experiments_run_mcts_till_end' and the scripts in 'scripts_test'

####Table 4 Comparison of MCTS: Classic vs. AmEx and Grammar: B vs. C
Switch to branch experiments_run_mcts_till_end_nguyen and use the scripts  in scripts 'scripts_nguyen.'
