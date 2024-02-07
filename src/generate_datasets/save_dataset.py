from pathlib import Path
from src.game.game_history import GameHistory
import pandas as pd
from src.game.find_equation_game import FindEquationGame
from src.utils.get_grammar import read_grammar_file
import numpy as np
import os
from pickle import Pickler, Unpickler, HIGHEST_PROTOCOL
from collections import deque

def save_panda_dataframes(save_folder, dict_measurements, approach, args):
    save_folder = Path(save_folder) / f'{approach}'
    save_folder.mkdir(exist_ok=True, parents=True)
    for key, values in dict_measurements.items():
        str_formula = f"{values['formula']}".replace('/', 'div')
        df = values['df']
        if args.store_multiple_versions_of_one_equation:
            df.to_csv(save_folder / f"{str_formula}__{key}.csv", index= False)
        else:
            df.to_csv(save_folder / f"{str_formula}.csv", index=False)

def save_supervise_buffer(args, path_to_save_buffer, dic_measurements):
    grammar = read_grammar_file(args=args)
    game = FindEquationGame(
        grammar,
        args,
        train_test_or_val='train'
    )

    buffer = deque(
            maxlen=args.selfplay_buffer_window)
    for measurement in dic_measurements.values():
        history = GameHistory()
        state = game.getInitialState()
        state.observation['data_frame']= measurement['df']
        state.observation['true_equation'] = measurement['formula']
        state.observation['true_equation_hash'] = measurement['formula']
        gamma =  0.99
        history.observed_returns = list()
        for i, action in enumerate(measurement['action_sequence']):
            state.action = action
            next_state, r = game.getNextState(
                state=state,
                action=action,
            )
            pi = np.zeros(game.action_size, dtype=np.float32)
            pi[action] = np.float32(1)
            v = np.float32(1)
            history.capture(
                state=state,
                pi=pi,
                r=r,
                v=v
            )
            history.observed_returns.append(1 * gamma**(len(measurement['action_sequence'])-i))
            history.found_equation = measurement['formula']

            state=next_state
        buffer.append([history])
    saveTrainExamples(buffer, path_to_save_buffer)


def saveTrainExamples(buffer, path_to_save_buffer) -> None:
    """
    Store the current accumulated data to a compressed file using pickle. Note that for highly dimensional
    environments, that the stored files may be considerably large and that storing/ loading the data may
    introduce a significant bottleneck to the runtime of the algorithm.
    :param iteration: int Current iteration of the self-play. Used as indexing value for the data filename.
    """

    if not os.path.exists(path_to_save_buffer):
        os.makedirs(path_to_save_buffer)
    filename = path_to_save_buffer / f"supervised.examples"
    with open(filename, "wb+") as f:
        Pickler(f, protocol=HIGHEST_PROTOCOL).dump(
            buffer)


