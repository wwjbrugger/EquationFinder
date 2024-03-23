import os
import glob
from pathlib import Path
import json
import datetime
import fnmatch

def run():

    wandb_entity_name = 'wwjbrugger'#'jgu-wandb'
    wandb_project = "23_03_train_model"
    Path_to_experiments = Path(
        "/home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/.wandb"
    )


    upload_experiment(Path_to_experiments, wandb_entity_name, wandb_project)


def upload_experiment(Path_to_experiments, wandb_entity_name, wandb_project):
    path_to_runs = get_path_to_wandb_file(Path_to_experiments)
    print(f"number of all runs: {len(path_to_runs)} ")
    i = 0
    for run in path_to_runs:
        print(f"iteration: {i}")
        i+=1
        if 'train_model' in run.parts[6]:
            try:
                command = f"wandb sync -e {wandb_entity_name} " \
                          f"-p {wandb_project} " \
                          f"--clean-force "\
                          f" {run}"
                print(command)
                os.system(command)
            except FileNotFoundError:
                print(f"skiped {run}")


def get_path_to_wandb_file(path_to_run_to_upload, paths = []):
    for child in path_to_run_to_upload.iterdir():
        if child.is_dir():
            if 'offline-run' in child.name:
                paths.append(child)
            else:
                paths = get_path_to_wandb_file(child, paths)
    return paths


if __name__ == '__main__':
    run()
