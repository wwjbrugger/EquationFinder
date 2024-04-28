import random
from pathlib import Path
import pandas as pd
import numpy as np
from definitions import ROOT_DIR
import tensorflow as tf
import tensorflow_text as tf_text
from src.preprocess_data.equation_preprocess_dummy import EquationPreprocessDummy
from pcfg import PCFG
from src.generate_datasets.grammars import get_grammars
from src.generate_datasets.dataset_generator import DatasetGenerator
class GenPandasPreprocess(EquationPreprocessDummy):
    """
    Class to read data dynamically to transformer model
    """

    def __init__(self, args, train_test_or_val):

        super().__init__(args, train_test_or_val)
        self.num_variables_in_grammar = self.get_num_variables_in_grammar(
            self.symbol_hash_dic
        )
        dataset_columns = [f"x_{i}" for i in range(self.num_variables_in_grammar)]
        dataset_columns.append('y')
        self.dataset_columns = dataset_columns
        pass

    def __str__(self):
        return 'GenPandasPreprocess'


    def get_num_variables_in_grammar(self, symbol_hash_dic):
        num_variables = len([key for key in symbol_hash_dic.keys() if key.startswith('x_') and key[2:].isdigit()])
        return num_variables



    def get_datasets(self):
        # returns an iterator
        self.num_production_rules = self.get_num_production_rules()
        iterator = GenPandasIterator(
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
        return'GenDataFrameReader'


class GenPandasIterator:

    def __init__(self, dataset_columns, args,
                 map_tree_representation_to_int):
        self.args = args
        self.dataset_columns = dataset_columns
        self.index = 0
        self.map_tree_representation_to_int = map_tree_representation_to_int
        self.generation_grammar = PCFG.fromstring(get_grammars(args))
        self.experiment_dataset_dic = {
            'num_calls_sampling': args.num_calls_sampling,
            'x_0': {
                'distribution': np.random.uniform,
                'distribution_args': {
                    'low': -5,
                    'high': 5,
                    'size': args.num_calls_sampling
                },
                'min_variable_range': 2,
                'generate_all_values_with_one_call': True,
                'sample_with_noise': args.sample_with_noise,
                'noise_std': 0.1
            },
            'x_1': {
                'distribution': np.random.uniform,
                'distribution_args': {
                    'low': -5,
                    'high': 5,
                    'size': args.num_calls_sampling
                },
                'min_variable_range': 2,
                'generate_all_values_with_one_call': True,
                'sample_with_noise': args.sample_with_noise,
                'noise_std': 0.1
            },
            'c': {
                'distribution': np.random.uniform,
                'distribution_args': {
                    'low': 0.5,
                    'high': 5,
                }
            }
        }
        self.dataset_generator = DatasetGenerator(
            grammar=self.generation_grammar,
            args=self.args,
            experiment_dataset_dic=self.experiment_dataset_dic
        )

    def __str__(self):
        return 'gen_pandas_iterator'

    def __iter__(self):
        return self

    def __next__(self):
        dic_measurements = self.dataset_generator.prepare_random_datasets()
        while not np.all(np.isfinite(dic_measurements[0]['df'].to_numpy())):
            dic_measurements = self.dataset_generator.prepare_random_datasets()
        original_columns = list(dic_measurements[0]['df'].columns)
        num_given_variables = len(original_columns) - 1
        dict_xi_to_variable_names = dict(zip(self.dataset_columns[:num_given_variables], original_columns))
        dict_variable_names_to_xi = dict(zip(original_columns, self.dataset_columns[:num_given_variables]))
        dict_variable_names_to_xi['target'] = 'y'
        for i, x_i in enumerate(self.dataset_columns[num_given_variables:-1]):
            dic_measurements[0]['df'].insert(i + num_given_variables, x_i, np.float32(0))
        dic_measurements[0]['df'].rename(columns=dict_variable_names_to_xi, inplace=True)
        shorten_data = dic_measurements[0]['df'].sample(
            n=min(dic_measurements[0]['df'].shape[0], self.args.max_len_datasets)
        )
        return {
            'formula': dic_measurements[0]['formula'],
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
