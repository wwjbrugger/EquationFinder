import tensorflow as tf
from definitions import ROOT_DIR
import numpy as np


def copy_dataset_encoder_weights_from_pretrained_agent(args, checkpoint_current_model, game):
    if args.path_to_pretrained_dataset_encoder:
        checkpoint_path_dataset_encoder = f"{ROOT_DIR / args.path_to_pretrained_dataset_encoder }"
        chkpoint_reader_dataset_encoder = tf.train.load_checkpoint(
            checkpoint_path_dataset_encoder
        )
        print(f"Copy weights of DatasetEncoder from {checkpoint_path_dataset_encoder}")
        selected_path = []
        for path in chkpoint_reader_dataset_encoder.get_variable_to_shape_map():
            if 'encoder_measurement' in path:
                selected_path.append(path)
        for selpath in selected_path:
            value = chkpoint_reader_dataset_encoder.get_tensor(selpath)
            assign_to_path(chkpoint_obj=checkpoint_current_model, path=selpath.split("/"), value=value)


def assign_to_path(chkpoint_obj, path: str, value):
    """
    recursively follow a checkpoint path read from a checkpoint and assign the variable in the appropriate leave
    """
    next_path = path[0]

    if next_path == ".ATTRIBUTES":
        chkpoint_obj.assign(value=value)
        return True
    elif next_path == ".OPTIMIZER_SLOT":
        raise NotImplementedError("assign_to_path::error::assignemnt of content of optimizer slots is not yet implemented.")
    elif next_path[0] in "0123456789":
        # path starting with a number is a list entry
        assign_to_path(chkpoint_obj=chkpoint_obj[int(next_path)], path=path[1:], value=value)
    elif hasattr(chkpoint_obj, next_path):
        assign_to_path(chkpoint_obj=getattr(chkpoint_obj, next_path), path=path[1:], value=value)
    else:
        raise RuntimeError(f"assign_to_path::error:: handling of path of form {next_path} with full path {path} not available")
