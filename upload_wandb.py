import os
import glob
from pathlib import Path
import json
import datetime
import fnmatch

def run():

    wandb_entity_name = 'wwjbrugger'#'jgu-wandb'
    wandb_project = "neural_guided_symbolic_regression_15_03"
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
    elements_in = glob.glob(f"{path_to_run_to_upload}/*")
    folder = [Path(x) for x in elements_in]
    folder = [Path(x) for x in folder if x.folder()]
    for f in folder:
        if 'offline-run' in f.name:
            paths.append(f)
        else:
            paths = get_path_to_wandb_file(f, paths=paths)
    return paths

    if 'offline-run' in name_file:
        paths.append()
    for folder in folder_in_wandb:
        name_folder = Path(folder).name
        files_in_wandb_folder = glob.glob(f"{path_to_run_to_upload}/{name_folder}/*")
        for file_in_wandb in files_in_wandb_folder:
            name_file = Path(file_in_wandb).name
            if 'offline-run' in name_file:
                path_to_run_to_upload = path_to_run_to_upload / name_file
                path_to_runs.append(file_in_wandb)
    return path_to_runs

def find_files(folder_path, pattern):
    result = []
    for root, dirs, files in os.walk(folder_path):
        for filename in fnmatch.filter(files, pattern):
            result.append(os.path.join(root, filename))
            for dir in dirs:
                result.appen(find_files())
    return result



def get_all_results(list_of_runs):
    return [Path(run_path).name for run_path in list_of_runs]


if __name__ == '__main__':
    run()
