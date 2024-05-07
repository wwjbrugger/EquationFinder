from src.generate_datasets.equation_generator import EquationGenerator
from argparse import ArgumentParser
from pcfg import PCFG
import random
import numpy as np
from src.generate_datasets.save_dataset import save_panda_dataframes
from src.generate_datasets.save_grammar_files import save_grammar_to_file
from pathlib import Path
from src.utils.parse_args import str2bool
from src.utils.files import create_file_path
from src.generate_datasets.dataset_generator import DatasetGenerator
from src.utils.get_grammar import get_grammar_from_string

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--save_folder", help="where to save data_ set",
                        required=True, type=str)
    parser.add_argument("--number_equations", default=10,
                        help="how many trees to generate", required=False,
                        type=int)
    parser.add_argument("--seed", default=42,
                        help="paths per tree", required=False, type=int)
    parser.add_argument("--max_depth_of_tree", default=6,
                        help="how many recursions a formula is allowed to have"
                        , required=False, type=int)
    parser.add_argument("--num_calls_sampling", default=50,
                        help="How often the sampling procedure is called per example",
                        required=False, type=int)

    parser.add_argument("--how_to_select_node_to_delete", type=str,
                        help="Choose int to delete this node id in all generated trees. "
                             "Choose 'all' to delete one node after each other"
                             "Choose random to delete one random node from each tree nodes "
                        )
    parser.add_argument("--sample_with_noise", type=str2bool,
                        help="Noise on x values "
                        )
    parser.add_argument("--logging_level", type=int, default=10,
                        help="CRITICAL = 50, ERROR = 40, "
                             "WARNING = 30, INFO = 20, "
                             "DEBUG = 10, NOTSET = 0")
    parser.add_argument('--max_branching_factor', type=float,
                        help='Estimate how many children a node will have at average')

    parser.add_argument('--store_multiple_versions_of_one_equation', type=str2bool, default=True,
                        help='If argument is true, each generated equation gets a unique identifier'
                             'If argument is false. identifier is missing and a new generated '
                             'equation will overwrite an existing formula which has the same string representation. ')

    parser.add_argument('--max_constants_in_tree', type=int, default=3,
                        help='Maximum number of constants allowed in  equation'
                             'afterwards equation will be invalid')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    grammar_string = \
        """  
        S -> '+' S S [0.1]
        S -> '-' S S [0.1]
        S -> '*' S S [0.1]
        S -> 'sin' Inner_Function [0.1] 
        S -> 'cos' Inner_Function [0.1] 
        S -> 'log' Inner_Function [0.1]  
        S -> 'x_0' [0.05] 
        S -> 'x_1' [0.05]
        S -> '**' Exponent Variable [0.1]
        S -> '1' [0.05]
        S -> '0.5' [0.05]
        S -> '2' [0.1]
        Exponent -> '6' [0.1]
        Exponent -> '5' [0.1] 
        Exponent -> '4' [0.1] 
        Exponent -> '3' [0.1] 
        Exponent -> '2' [0.2] 
        Exponent -> '0.5' [0.2] 
        Exponent -> 'x_1' [0.2]
        Inner_Function -> '**' Exponent Variable [0.3] 
        Inner_Function -> 'x_0' [0.2] 
        Inner_Function -> 'x_1' [0.2] 
        Inner_Function -> '+' SUM SUM [0.3]  
        SUM -> '**' Exponent Variable [0.5] |
        SUM -> '1' [0.2] 
        SUM -> 'x_0' [0.15] 
        SUM -> 'x_1' [0.15]
        Variable -> 'x_0' [0.5] | 'x_1' [0.5]
           """
    grammar = get_grammar_from_string(grammar_string)

    actions_list = [[2, 11, 2, 3, 20, 4, 21]]
    experiment_dataset_dic = {
        'num_calls_sampling': args.num_calls_sampling,
        'x_0': {
            'distribution': np.random.uniform,
            'distribution_args': {
                'low': 0,
                'high': 10,
                'size': args.num_calls_sampling
            },
            'generate_all_values_with_one_call': True,
            'sample_with_noise': args.sample_with_noise,
            'noise_std': 0.1
        },
        'x_1': {
            'distribution': np.random.uniform,
            'distribution_args': {
                'low': 0,
                'high': 10,
                'size': args.num_calls_sampling
            },
            'generate_all_values_with_one_call': True,
            'sample_with_noise': args.sample_with_noise,
            'noise_std': 0.1
        }

    }

    save_folder = Path(args.save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)
    save_path = create_file_path(save_folder=save_folder, stem='run')

    dataset_generator = DatasetGenerator(
        grammar=grammar,
        args=args,
        experiment_dataset_dic=experiment_dataset_dic
    )

    dataset_generator.save_production_rules_to_file(save_path)
    dataset_generator.save_used_symbols_to_file(save_path, additional_symbols=[])

    for data_type in ['train']:  # , 'val', 'test']:
        dic_measurements = dataset_generator.prepare_datasets_from_actions(
            actions_list=actions_list
        )

    save_panda_dataframes(
        save_folder=save_path / data_type,
        dict_measurements=dic_measurements,
        approach='pandas',
        args=args
    )
