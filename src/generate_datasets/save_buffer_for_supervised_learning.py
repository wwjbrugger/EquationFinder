from definitions import ROOT_DIR
from src.generate_datasets.save_dataset import save_supervise_buffer
from pathlib import Path
def save_buffer_for_supervised_learning(args, save_path, dic_measurements):
    args.data_path = f"{save_path.parent.name}/{save_path.name}"
    args.prior_source = 'None'
    args.max_elements_in_best_list = 1
    args.max_len_datasets = 2
    args.max_tokens_equation = 64
    args.minimum_reward = -1
    args.selfplay_buffer_window = 10000
    path_to_save_buffer = f'{ROOT_DIR}/saved_models/{save_path.parent.name}/{save_path.name}/supervised_buffer'
    save_supervise_buffer(
        args=args,
        path_to_save_buffer=Path(path_to_save_buffer),
        dic_measurements=dic_measurements
    )
    print(f"supervised_buffer is saved to {path_to_save_buffer}")