import random
from pathlib import Path
import pandas as pd
import numpy as np
from definitions import ROOT_DIR
import tensorflow as tf
import tensorflow_text as tf_text
from src.preprocess_data.equation_preprocess_dummy import EquationPreprocessDummy

class PandasPreprocess(EquationPreprocessDummy):
    """
    Class to read data dynamically to transformer model
    """

    def __init__(self, args, train_test_or_val, grammar):
        self.grammar = grammar

        super().__init__(args, train_test_or_val, self.grammar)
        self.num_variables_in_grammar = self.get_num_variables_in_grammar(
            self.symbol_hash_dic
        )
        dataset_columns = [f"x_{i}" for i in range(self.num_variables_in_grammar)]
        dataset_columns.append('y')
        self.dataset_columns = dataset_columns
        pass

    def __str__(self):
        return 'PandasPreprocess'


    def get_num_variables_in_grammar(self, symbol_hash_dic):
        num_variables = len([key for key in symbol_hash_dic.keys() if key.startswith('x_') and key[2:].isdigit()])
        return num_variables



    def get_datasets(self):
        # returns an iterator
        self.num_production_rules = self.get_num_production_rules()
        iterator = PandasIterator(
            path_to_panda_frames=ROOT_DIR / self.args.data_path /
                                 self.train_test_or_val / 'pandas',
            args=self.args,
            dataset_columns=self.dataset_columns,
            map_tree_representation_to_int=self.map_tree_representation_to_int
        )
        return iterator

    def preprocess(self, dataset):
        raise ImportError('This mathod can not be deleted' )
        dataset = self.add_int_rep_of_tree(dataset)
        dataset = self.split_production_index(dataset)
        return dataset

    def __str__(self):
        return'DataFrameReader'


class PandasIterator:
    def __init__(self, path_to_panda_frames, dataset_columns, args,
                 map_tree_representation_to_int):
        self.args = args
        self.dataset_columns = dataset_columns
        self.list_path_to_frames = []
        self.index = 0
        p = Path(f"{path_to_panda_frames}").glob('**/*')
        self.files = [x for x in p if x.is_file()]
        random.shuffle(self.files)
        self.num_datasets = len(self.files)
        self.map_tree_representation_to_int = map_tree_representation_to_int

    def __str__(self):
        return 'pandas_iterator'

    def __iter__(self):
        return self

    def __next__(self):
        input_data, path = self.get_frame_from_disc()
        while not np.all(np.isfinite(input_data.to_numpy())):
            print('This should not happen. On of the input data has an non finite element.#'
                  'Load a new one ')
            input_data, path = self.get_frame_from_disc()
        original_columns = list(input_data.columns)
        num_given_variables = len(original_columns) - 1
        dict_xi_to_variable_names = dict(zip(self.dataset_columns[:num_given_variables], original_columns))
        dict_variable_names_to_xi = dict(zip(original_columns, self.dataset_columns[:num_given_variables]))
        dict_variable_names_to_xi['target'] = 'y'
        for i, x_i in enumerate(self.dataset_columns[num_given_variables:-1]):
            input_data.insert(i + num_given_variables, x_i, np.float32(0))
        input_data.rename(columns=dict_variable_names_to_xi, inplace=True)
        shorten_data = input_data.sample(n=min(input_data.shape[0], self.args.max_len_datasets))
        return {
            'infix_formula': path.stem,
            'data_frame': shorten_data,
            'dict_xi_to_variable_names': dict_xi_to_variable_names
        }

    def get_frame_from_disc(self):
        path = self.files[self.index]
        self.index = (self.index + 1) % self.num_datasets
        input_data, self.feature_names = read_file(path)
        return input_data, path


def read_file(filename, label='target', sep=None):
    if filename.suffix == '.gz':
        compression = 'gzip'
    else:
        compression = None
    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression, dtype=np.float32)
    else:
        input_data = pd.read_csv(filename, sep=sep, compression=compression,
                                 engine='python', dtype=np.float32)
    feature_names = input_data.columns.values
    return input_data, feature_names
