import unittest
import random
from pcfg import PCFG
import numpy as np
from src.generate_datasets.dataset_generator import DatasetGenerator
import pickle as pkl
from definitions import ROOT_DIR
from src.utils.get_grammar import get_grammar_from_string


class TestEquationGenerator(unittest.TestCase):
    def setUp(self) -> None:
        grammar_string = \
            """         S -> Basic [0.7]
                        S -> Advanced [0.3]

                        Basic -> '*' S S [0.2]
                        Basic -> '/' S S [0.2]
                        Basic -> '+' S S [0.2]
                        Basic -> '-' S S [0.2]
                        Basic -> Variable  [0.1]
                        Basic -> '**' '2' Variable [0.1]

                        Advanced -> 'c'   Basic [0.2]
                        Advanced -> 'exp' Basic [0.2]
                        Advanced -> '**'  'c' Basic [0.2]
                        Advanced -> 'log' Basic [0.2]
                        Advanced -> 'sin' Basic [0.1]
                        Advanced -> 'cos' Basic [0.1] 

                        Variable -> 'x_0'[0.5]
                        Variable -> 'x_1'[0.5]

               """

        class Namespace():
            def __init__(self):
                pass

        self.args = Namespace()
        self.args.logging_level = 40
        self.args.max_branching_factor = 2
        self.args.max_depth_of_tree = 11
        self.args.number_equations = 10
        self.args.num_calls_sampling = 10
        self.args.sample_with_noise = False
        self.args.how_to_select_node_to_delete = 'random'
        self.args.max_constants_in_tree = 3
        experiment_dataset_dic = experiment_dataset_dic = {
            'num_calls_sampling': self.args.num_calls_sampling,
            'x_0': {
                'distribution': np.random.uniform,
                'distribution_args': {
                    'low': -10,
                    'high': 10,
                    'size': self.args.num_calls_sampling
                },
                'generate_all_values_with_one_call': True,
                'sample_with_noise': self.args.sample_with_noise,
                'noise_std': 0.1
            },
            'x_1': {
                'distribution': np.random.uniform,
                'distribution_args': {
                    'low': -10,
                    'high': 10,
                    'size': self.args.num_calls_sampling
                },
                'generate_all_values_with_one_call': True,
                'sample_with_noise': self.args.sample_with_noise,
                'noise_std': 0.1
            },
            'c': {
                'distribution': np.random.uniform,
                'distribution_args': {
                    'low': -10,
                    'high': 10,
                }
            }
        }

        grammar = get_grammar_from_string(grammar_string)
        self.generator = DatasetGenerator(
            grammar=grammar,
            args=self.args,
            experiment_dataset_dic=experiment_dataset_dic)

    # def test_save_dataset(self):
    #     random.seed(42)
    #     dict_measurements = self.generator.prepare_random_datasets()
    #     with open(ROOT_DIR / 'test_case' / 'own_grammar_parser' / 'saved_object' / 'generated_dataset.obj', 'wb') as handle:
    #         pkl.dump(dict_measurements, handle, protocol=pkl.HIGHEST_PROTOCOL)


    def test_load_dataset(self):
        random.seed(42)
        dict_measurements = self.generator.prepare_random_datasets()

        with open(ROOT_DIR / 'test' / 'saved_object' / 'generated_dataset.obj', 'rb') as handle:
            dict_measurements_load = pkl.load(handle)
        self.assertEqual(dict_measurements[0]['formula'], dict_measurements_load[0]['formula'])

