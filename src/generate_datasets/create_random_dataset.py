from src.generate_datasets.equation_generator import EquationGenerator
from argparse import ArgumentParser
from pcfg import PCFG
import random
import numpy as np
from src.generate_datasets.save_dataset import save_panda_dataframes
from src.generate_datasets.grammars import get_grammars
from src.generate_datasets.save_grammar_files import save_grammar_to_file
from pathlib import Path
from src.utils.parse_args import str2bool
from src.utils.files import create_file_path
from src.generate_datasets.dataset_generator import DatasetGenerator
from src.generate_datasets.split_dataset import split_dataset
from definitions import ROOT_DIR
from src.generate_datasets.save_buffer_for_supervised_learning import save_buffer_for_supervised_learning





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--save_folder", help="where to save data_ set",
                        required=True, type=str)
    parser.add_argument("--grammar_to_use", default=True)
    parser.add_argument("--number_equations", default=2000,
                        help="how many trees to generate", required=False,
                        type=int)
    parser.add_argument("--max_number_equation_of_one_type", default=10,
                        help="For each syntax tree multiple constants and x values can be sampled. "
                             "This argument gives an upper bound on how often a syntax tree can be in the dataset.", required=False,
                        type=int)
    parser.add_argument("--seed", default=42,
                        help="seed to generate trees", required=False, type=int)
    parser.add_argument("--max_depth_of_tree", default=10,
                        help="how many recursions a formula is allowed to have"
                        , required=False, type=int)
    parser.add_argument('--max_num_nodes_in_syntax_tree', type=int,
                        help='Maximum nodes of generated equations', default=25)
    parser.add_argument("--num_calls_sampling", default=20,
                        help="How often the sampling procedure is called per example",
                        required=False, type=int)
    parser.add_argument("--how_to_select_node_to_delete", default=0,
                        type=str,
                        help="Choose int to delete this node id in all generated trees. "
                             "Choose 'all' to delete one node after each other"
                             "Choose random to delete one random node from each tree nodes "
                        )
    parser.add_argument("--sample_with_noise", default=False,
                        type=str2bool,
                        help="Noise on x values "
                        )
    parser.add_argument("--logging_level", type=int, default=30,
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
    experiment_dataset_dic = {
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


    grammar = PCFG.fromstring(get_grammars(args))

    save_folder = ROOT_DIR / Path(args.save_folder)
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
        dic_measurements = dataset_generator.prepare_random_datasets()

    save_panda_dataframes(
        save_folder=save_path / data_type,
        dict_measurements=dic_measurements,
        approach='pandas',
        args=args
    )

    split_dataset(path=Path(f'{ROOT_DIR}/{save_path.parent.name}/{save_path.name}'))
    print(f"Datasets are saved to  {ROOT_DIR}/{save_path.parent.name}/{save_path.name}")

    save_buffer_for_supervised_learning(
        args=args,
        save_path = save_path,
        dic_measurements=dic_measurements
    )
