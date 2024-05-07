import copy

from src.generate_datasets.equation_generator import EquationGenerator
from nltk.grammar import Nonterminal
import numpy as np
import pandas as pd
import random
from src.utils.error import NonFiniteError
from src.syntax_tree.syntax_tree import SyntaxTree
class DatasetGenerator():
    def __init__(self, grammar, args, experiment_dataset_dic):
        self.grammar = grammar
        self.args = args
        self.equation_generator = EquationGenerator(grammar=self.grammar,
                                                    args=self.args)
        self.experiment_dataset_dic = experiment_dataset_dic
        self.how_often_equation_generated = {}

    def save_production_rules_to_file(self, save_folder):
        save_folder.mkdir(exist_ok=True, parents=True)
        with open(save_folder / 'production_rules.txt', "a") as file:
            file.writelines(str(production) + '\n' for production in
                            self.grammar._productions)

    def update_how_often_equation_generated(self,new_equation):
        if not new_equation.__str__() in self.how_often_equation_generated:
            self.how_often_equation_generated[new_equation.__str__()] = 1
        else:
            self.how_often_equation_generated[new_equation.__str__()] += 1

    def equation_type_over_limit(self, new_equation):
        if not new_equation.__str__() in self.how_often_equation_generated:
            return False

        if  (self.how_often_equation_generated[new_equation.__str__()]
                < self.args.max_number_equation_of_one_type):
            return False
        else:
            return True

    def save_grammar_to_file(self, save_folder):
        save_folder.mkdir(exist_ok=True, parents=True)
        with open(save_folder / 'grammar.txt', "a") as file:
            file.writelines(line for line in self.grammar)

    def prepare_random_datasets(self):
        num_sampled_equations = 0
        dic_measurements = {}

        while num_sampled_equations < self.args.number_equations:
            new_equation, action_sequence = self.equation_generator.create_new_equation()
            i = 0
            while ((not new_equation.complete)
                   or self.equation_type_over_limit(new_equation)):
                new_equation, action_sequence = self.equation_generator.create_new_equation()
                i += 1
                if i % 500 == 498:
                    print("Generation of 500 equations were not successful.")

            try:
                df = self.create_experiment_dataset(equation=new_equation)
                full_equation_string = new_equation.rearrange_equation_infix_notation(-1)[1]
                s = constant_dict_to_string(new_equation)

                dic_measurements[num_sampled_equations] = {
                    'formula': full_equation_string + s,
                    'df': df,
                    'action_sequence': action_sequence
                }
                num_sampled_equations += 1
                self.update_how_often_equation_generated(new_equation)
            except OverflowError as e:
                print(f'Tree was not successfully created : {e}')
            except ZeroDivisionError as e:
                print(f'Equation is generated in which a division with 0  has be applied: {e} ')
            except FloatingPointError as e:
                print(f'Equation is generated in which {e}')
            except NonFiniteError:
                print('Non finite element in y_calc')

            if num_sampled_equations % 500 == 0:
                print(f"{num_sampled_equations} Trees are generated from {self.args.number_equations}")

        return dic_measurements

    def prepare_datasets_from_actions(self, actions_list):
        num_sampled_equations = 0
        dic_measurements = {}
        for actions in actions_list:
            new_equation = SyntaxTree(grammar=self.grammar, args=self.args)
            for action in actions:
                node_to_expand = new_equation.nodes_to_expand[0]
                new_equation.expand_node_with_action(
                    node_id=node_to_expand,
                    action=action,
                    build_syntax_tree_eager = True

                )
            try:
                df = self.create_experiment_dataset(equation=new_equation)
                full_equation_string = new_equation.rearrange_equation_infix_notation(-1)[1]
                s = constant_dict_to_string(new_equation)

                dic_measurements[num_sampled_equations] = {
                    'formula': full_equation_string + s,
                    'df': df
                }
                num_sampled_equations += 1
            except OverflowError as e:
                print(f'Tree was not successfully created : {e}')
            except ZeroDivisionError as e:
                print(f'Equation is generated in which a division with 0  has be applied: {e} ')
            except FloatingPointError as e:
                print(f'Equation is generated in which {e}')
            except NonFiniteError:
                print('Non finite element in y_calc')

            if num_sampled_equations % 500 == 0:
                print(f"{num_sampled_equations} Trees are generated from {self.args.number_equations}")

        return dic_measurements



    def create_experiment_dataset(self, equation):
        """
        create dataset with experimental data
        :param experiment_dataset_dic: Dict with information about the distribution of variables
        :return:
        """
        data_frame = self.generate_values_for_variables(equation=equation)
        equation = self.generate_values_for_constants(equation=equation)
        data_frame = self.add_equation_result_to_df(data_frame, equation)
        return data_frame

    def generate_values_for_variables(self, equation):
        """
        sample values for variables
        :param variables:
        :param experiment_dataset_dic: Dict with information about the distribution of variables
        :return:
        """
        variables = self.grammar._leftcorner_words[Nonterminal('Variable')]
        num_calls_sampling = self.experiment_dataset_dic['num_calls_sampling']
        measurement_dict = {}
        sorted_variables = list(variables)
        sorted_variables.sort()
        for variable in sorted_variables:
            distribution = self.experiment_dataset_dic[variable]['distribution']
            distribution_args = self.experiment_dataset_dic[variable]['distribution_args']

            new_distribution_args = self.sample_new_boarder_values(
                distribution_args,
                variable = variable,
                min_variable_range = self.experiment_dataset_dic[variable]['min_variable_range'],
                equation=equation
            )
            if self.experiment_dataset_dic[variable]['generate_all_values_with_one_call']:
                measurements = [value for value in distribution(**new_distribution_args)[:num_calls_sampling]]
            else:
                measurements = [distribution(**new_distribution_args) for _ in range(num_calls_sampling)]
            if self.experiment_dataset_dic[variable]['sample_with_noise']:
                measurements = np.random.normal(
                    measurements, self.experiment_dataset_dic[variable]['noise_std'])
            measurement_dict[variable] = measurements
        return pd.DataFrame(measurement_dict)
    
    def sample_new_boarder_values(self, distribution_args, variable, min_variable_range, equation):
        new_distribution_args = copy.deepcopy(distribution_args)
        self.get_boarders_of_equation(equation, new_distribution_args, variable)
        new_distribution_args = self.sample_range_in_allowed_range(
            min_variable_range=min_variable_range,
            new_distribution_args=new_distribution_args,
            variable=variable
        )
        return new_distribution_args

    def sample_range_in_allowed_range(self, min_variable_range, new_distribution_args, variable):
        if self.experiment_dataset_dic[variable]['min_variable_range'] < new_distribution_args['high'] - new_distribution_args['low']:
            new_data_range = 0
            while new_data_range < min_variable_range:
                new_low, new_high = np.sort(np.random.uniform(
                    low=new_distribution_args['low'],
                    high=new_distribution_args['high'],
                    size=2
                ))
                new_data_range = new_high - new_low
            new_distribution_args['low'] = new_low
            new_distribution_args['high'] = new_high
        return new_distribution_args

    def get_boarders_of_equation(self, equation, new_distribution_args, variable):
        v_low, v_high = equation.operators_data_range(variable)
        if v_low > new_distribution_args['high'] or v_high < new_distribution_args['low']:
            raise AssertionError(f"new boarders are outside of the distribution parameter")
        if v_low > new_distribution_args['low']:
            new_distribution_args['low'] = v_low
        if v_high < new_distribution_args['high']:
            new_distribution_args['high'] = v_high


    def generate_values_for_constants(self, equation):
        if 'c' in self.experiment_dataset_dic:
            distribution = self.experiment_dataset_dic['c']['distribution']
            distribution_args = self.experiment_dataset_dic['c']['distribution_args']
            for key, value in equation.constants_in_tree.items():
                if key != 'num_fitted_constants':
                    sampled_constant = distribution(**distribution_args)
                    sampled_constant = np.round(sampled_constant, decimals=2)
                    value['value'] = sampled_constant
                    equation.constants_in_tree['num_fitted_constants'] +=1
        return equation


    def add_equation_result_to_df(self, data_frame, equation):
        """
        evaluate created formula on sampled values
        :param data_frame:
        :param str_representation:
        :return:
        """
        y_list = []
        _, string = equation.rearrange_equation_infix_notation(new_start_node_id=-1)
        y = equation.evaluate_subtree(node_id=0, dataset=data_frame)

        data_frame['y'] = y
        return data_frame

    def select_nodes_to_delete(self, equation, how_to_select_node_to_delete):
        list_of_nodes = list(equation.dict_of_nodes.keys())
        self.remove_y_node_from_deletable_nodes(list_of_nodes)
        if how_to_select_node_to_delete == 'random':
            list_id_last_node = random.sample(list_of_nodes, 1)
        elif how_to_select_node_to_delete == 'all':
            list_id_last_node = list_of_nodes
        else:
            list_id_last_node = [int(how_to_select_node_to_delete)]
        list_id_last_node.sort(reverse=True)
        return list_id_last_node

    def remove_y_node_from_deletable_nodes(self, list_of_nodes):
        # removes start and y node
        list_of_nodes.remove(-1)

    def fill_dict_tree(self, label, equation, num_sampled_equations, id_last_node, last_symbol, dict_tree):
        """
        Prepare dict to save trees in a format  which include all outer nodes
        :param last_symbol:
        :param label:
        :param dict_whole_tree_str:
        :param num_sampled_rows: How many measurements per experiment should be done
        :return:
        """
        _, infix_notion = equation.rearrange_equation_infix_notation(new_start_node_id=-1)
        _, prefix_notion = equation.rearrange_equation_prefix_notation(new_start_node_id=-1)
        dict_tree[num_sampled_equations] = {}
        dict_tree[num_sampled_equations]['string'] = infix_notion
        dict_tree[num_sampled_equations]['label'] = label
        dict_tree[num_sampled_equations]['id_last_node'] = id_last_node
        dict_tree[num_sampled_equations]['last_symbol'] = last_symbol
        dict_tree[num_sampled_equations]['production_index'] = prefix_notion

    def save_used_symbols_to_file(self, save_folder, additional_symbols):
        save_folder.mkdir(exist_ok=True, parents=True)
        all_symbols_from_grammar = self.get_all_symbols_usable()
        all_symbols = all_symbols_from_grammar.union(set(additional_symbols))
        all_symbols = [str(symbol) for symbol in all_symbols]
        all_symbols.sort()
        with open(save_folder / 'symbols.txt', "a") as file:
            file.writelines(str(symbol) + ', ' for symbol in all_symbols)

    def get_all_symbols_usable(self):
        terminal_symbols = set(self.grammar._lexical_index.keys())
        non_terminal_symbols = set(self.grammar._lhs_index.keys())
        all_symbols = terminal_symbols.union(non_terminal_symbols)
        return all_symbols

def constant_dict_to_string(new_equation):
    s = ''
    for key, value in new_equation.constants_in_tree.items():
        if key != 'num_fitted_constants':
            s += key
            s += f"_{str(value['value'])}_"
    return s
