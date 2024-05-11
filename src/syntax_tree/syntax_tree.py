import traceback

import tqdm

from src.equation_classes.math_class.plus import Plus
from src.equation_classes.node import Node
from src.equation_classes.math_class.terminal import Terminal
from src.equation_classes.math_class.division import Division
from src.equation_classes.math_class.minus import Minus
from src.equation_classes.math_class.multiplication import Multiplication
from src.equation_classes.math_class.sine import Sine
from src.equation_classes.math_class.cosine import Cosine
from src.equation_classes.math_class.y import Y
from src.equation_classes.math_class.power import Power
from src.equation_classes.math_class.logarithm import Logarithm
from src.equation_classes.math_class.constants import Constants
from src.equation_classes.math_class.logarithmus_naturalis import Logarithm_naturalis
from src.utils.logging import get_log_obj
import numpy as np
import bisect
from src.constant_fitting.helper_object_constant_fitting import HelperObjectConstantFitting
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning
from src.equation_classes.math_class.exp import Exp
from src.utils.error import NonFiniteError
import copy


class SyntaxTree():
    """
    Class to represent equations as tree
    """

    def __init__(self, grammar, args):
        self.logger = get_log_obj(args=args, name='SyntaxTree')
        np.seterr(all='raise')
        self.operator_to_class = {
            '+': Plus,
            'terminal': Terminal,
            '/': Division,
            '-': Minus,
            '*': Multiplication,
            'sin': Sine,
            'cos': Cosine,
            'y': Y,
            '**': Power,
            'ln': Logarithm_naturalis,
            'log': Logarithm,
            'c': Constants,
            'exp': Exp
        }
        self.grammar = grammar
        self.args = args
        self.nodes_to_expand = []
        self.dict_of_nodes = {}
        self.max_depth = args.max_depth_of_tree

        self.current_depth = 0  # todo gets not updated, if child is deleted
        self.complete = False
        self.max_depth_reached = False
        self.max_constants_reached = False
        self.max_nodes_reached = False
        self.max_nodes_allowed = self.args.max_num_nodes_in_syntax_tree
        self.non_terminals = []
        self.max_branching_factor = args.max_branching_factor
        self.add_start_node()
        self.constants_in_tree = {
            'num_fitted_constants': 0
        }
        self.num_constants_in_complete_tree = 0
        self.action_buffer = []
        self.invalid = False

        if not grammar is None:
            self.start_node.node_symbol = str(grammar._start)
            self.start_node.invertible = True
            self.start_node.math_class = self.operator_to_class['terminal'](
                node=self.start_node
            )
            self.possible_products_for_symbol = {}
            self.fill_dict_possible_productions_for_symbol()
            self.non_terminals = [str(symbol) for symbol in set(self.grammar._lhs_index.keys())]

    def fill_dict_possible_productions_for_symbol(self):
        """
        Iteration through grammar rules and build a dict with possible rules
        for each symbol
        :return:
        """
        for i, production in enumerate(self.grammar._productions):
            if str(production._lhs) in self.possible_products_for_symbol.keys():
                self.possible_products_for_symbol[str(production._lhs)].append(i)
            else:
                self.possible_products_for_symbol[str(production._lhs)] = [i]
        return

    def fill_dict_for_symbol_to_productions(self):
        """
        Iteration through grammar rules and build a dict with possible rules
        for each symbol
        :return:
        """
        self.symbol_to_product = {}
        for i, production in enumerate(self.grammar._productions):
            if str(production._rhs[0]) in self.symbol_to_product.keys():
                self.symbol_to_product[str(production._rhs[0])].append(i)
            else:
                self.symbol_to_product[str(production._rhs[0])] = [i]
        return

    def possible_production_for_node(self, parent_node_id):
        node_symbol = self.dict_of_nodes[parent_node_id].node_symbol
        action_sequence = []
        if node_symbol in self.symbol_to_product:
            action_sequence = [self.symbol_to_product[node_symbol]]
            for child_node in self.dict_of_nodes[parent_node_id].list_children:
                child_action_sequence = self.possible_production_for_node(child_node.node_id)
                [action_sequence.append(child_action) for child_action in child_action_sequence]
        return action_sequence

    def possible_production_for_tree(self):
        equation_str = self.rearrange_equation_prefix_notation(new_start_node_id=-1)[1].replace(' ', '')
        self.fill_dict_for_symbol_to_productions()
        productions = self.possible_production_for_node(parent_node_id=0)
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        self.possible_production_for_tree_list = []
        self._possible_production_for_tree(productions, syntax_tree=syntax_tree,
                                           true_equation_str=equation_str)
        return self.possible_production_for_tree_list

    def _possible_production_for_tree(self, productions, syntax_tree, true_equation_str,
                                      i=0, action_sequence=[]):
        for action in productions[i]:
            try:
                node_to_expand = syntax_tree.nodes_to_expand[0]
                syntax_tree.expand_node_with_action(
                    node_id=node_to_expand,
                    action=action,
                    build_syntax_tree_token_based=False
                )
                action_sequence_to_expand = copy.deepcopy(action_sequence)
                action_sequence_to_expand.append(action)
                if i < len(productions) - 1:
                    self._possible_production_for_tree(
                        productions,
                        syntax_tree=syntax_tree,
                        i=i + 1,
                        action_sequence=action_sequence_to_expand,
                        true_equation_str=true_equation_str
                    )
                else:
                    if true_equation_str in syntax_tree.__str__().replace(' ', ''):
                        self.possible_production_for_tree_list.append((action_sequence_to_expand, syntax_tree.__str__()))
                    print(f"{f'sequence: {action_sequence_to_expand},':<100} {syntax_tree.__str__()}")
                syntax_tree.delete_children(node_id=node_to_expand, step_wise=False)
            except ValueError as e:
                pass
            except IndexError as e:
                pass

    def add_start_node(self):
        """
        Add an node with id 0 without mathematical class.
        Its parent node is a y node.
        Only important if the equation should be rearranged
        :return:
        """
        parent_node = Node(
            tree=self,
            parent_node=None,
            node_id=-1,
            depth=-1
        )
        parent_node.node_symbol = 'y'
        parent_node.math_class = self.operator_to_class['y'](
            node=parent_node
        )
        parent_node.invertible = True
        self.start_node = Node(
            tree=self,
            parent_node=parent_node,
            node_id=0,
            depth=0)
        parent_node.list_children.append(self.start_node)
        self.nodes_to_expand.remove(-1)

    def prefix_to_syntax_tree(self, prefix):
        """
        Construct from a string in prefix order a syntax tree.
        :param prefix:
        :return:
        """
        prefix_rest = self.start_node.prefix_to_syntax_tree(prefix)
        if len(prefix) > 0:
            raise SyntaxError(f'Not the complete prefix is translated to an syntax tree. The rest is : {prefix_rest}')
        if len(self.nodes_to_expand) == 0:
            self.complete = True

    def print(self):
        """
        Print Syntax tree in a nice way where the different depth of the tree is shown
        :return:
        """
        self.start_node.print()

    def count_nodes_in_tree(self):
        """
        Count the number of nodes in the tree
        :return:
        """
        i = self.start_node.count_nodes_in_tree()
        i += 1  # For the y node
        return i

    def delete_children(self, node_id, step_wise):
        """
        Delete the child nodes of a tree
        :param node_id:  Node which child should be deleted
        :param step_wise: A Node can have several productions which
        lead to the final symbol e.g. S -> Variable -> x_0
        if true the selected_production of the node is only shoten by one element.
        :return:
        """
        node = self.dict_of_nodes[node_id]
        while len(node.list_children) > 0:
            node.list_children[0].delete()
        if not node_id in self.nodes_to_expand:
            bisect.insort(self.nodes_to_expand, node_id)
        if len(node.list_children) > 0:
            raise AssertionError(f"After deleting all children there should"
                                 f" be no nodes in list_children"
                                 f" but there are {node.node_children}")
        last_symbol = node.node_symbol

        if len(node.selected_production) == 0:
            return '', None
        elif step_wise:
            production_index = node.selected_action[-1]
            node.node_symbol = str(node.selected_production[-1]._lhs)
            node.selected_production = node.selected_production[:-1]
            node.selected_action = node.selected_action[:-1]
            node.math_class = self.operator_to_class['terminal'](
                node=node
            )
            return last_symbol, production_index
        else:
            production_index = node.selected_action[-1]
            node.node_symbol = str(node.selected_production[0]._lhs)
            node.selected_production = []
            node.selected_action = []
            node.math_class = self.operator_to_class['terminal'](
                node=node
            )
            return last_symbol, production_index

    def rearrange_equation_prefix_notation(self, new_start_node_id=-1):
        """
        Returns the equation string rearranged to new_start_node_id in prefix notion
        :param new_start_node_id:
        :return:
        """
        new_start_node = self.dict_of_nodes[new_start_node_id]
        if len(self.action_buffer) > 0:
            return new_start_node.node_symbol, self.action_buffer_to_string()
        elif self.invalid:
            return new_start_node.node_symbol, '- - -'
        if new_start_node.invertible:
            equation = new_start_node.math_class.prefix_notation(
                call_node_id=new_start_node_id,
                kwargs={}
            )
            return new_start_node.node_symbol, equation
        else:
            raise AssertionError(f'Node {new_start_node_id} is not invertible')

    def get_subtree_in_prefix_notion(self, node_id):
        """
        Get the subtree of a node in prefix notion
        :param node_id:
        :return:
        """
        node = self.dict_of_nodes[node_id]
        parent_id = node.parent_node.node_id
        subtree_in_prefix = node.math_class.prefix_notation(
            call_node_id=parent_id
        )
        return subtree_in_prefix

    def rearrange_equation_infix_notation(self, new_start_node_id, kwargs={}):
        """
        Returns the equation string rearranged to new_start_node_id in infix notion
        :param new_start_node_id:
        :param kwargs:
        :return:
        """
        new_start_node = self.dict_of_nodes[new_start_node_id]
        if len(self.action_buffer) > 0:
            return new_start_node.node_symbol, self.action_buffer_to_string()
        elif self.invalid:
            return new_start_node.node_symbol, '- - -'
        if new_start_node.invertible:
            equation = new_start_node.math_class.infix_notation(
                new_start_node_id,
                kwargs
            )
            return new_start_node.node_symbol, equation
        else:
            raise AssertionError(f'Node {new_start_node_id} is not invertible')

    def expand_node_with_action(self, node_id, action,
                                build_syntax_tree_token_based=False):
        if build_syntax_tree_token_based == True:
            self.expand_node_with_token(action)
        else:
            node = self.dict_of_nodes[node_id]
            node.expand_node_with_action(action=action)
            if len(self.nodes_to_expand) == 0:
                self.complete = True


    def expand_node_with_token(self, action):
        if (action == self.grammar.terminal_action
                or len(self.action_buffer) > self.max_nodes_allowed):
            try:
                prefix = []
                for action in self.action_buffer:
                    production = self.grammar._productions[action]
                    rhs = copy.deepcopy(list(production.rhs()))
                    for symbol in rhs:
                        symbol = str(symbol)
                        if symbol != 'S':
                            prefix.append(symbol)
                self.prefix_to_syntax_tree(prefix=prefix)
                if len(self.nodes_to_expand) == 0:
                    self.complete = True
                else:
                    self.invalid = True
            except:
                self.invalid=True
            self.action_buffer = []
        else:
            self.action_buffer.append(action)

    def action_buffer_to_string(self):
        string_representation = ''
        for i, action in enumerate(self.action_buffer):
            production = self.grammar._productions[action]
            for token in production._rhs:
                if str(token) != 'S' or i == len(self.action_buffer)-1:
                    string_representation += f" {str(token)}"
        return string_representation

    def get_possible_moves(self, node_id):
        symbol = self.dict_of_nodes[node_id].node_symbol
        try:
            possible_moves = self.possible_products_for_symbol[str(symbol)]
        except KeyError:
            self.logger.error(f'In Equation {self.print()} an error occur '
                              f'nodes to expand are {self.nodes_to_expand}')

        return possible_moves

    def evaluate_subtree(self, node_id, dataset, return_equation_string=False):
        node_to_evaluate = self.dict_of_nodes[node_id]
        result = node_to_evaluate.math_class.evaluate_subtree(call_node_id=node_id,
                                                              dataset=dataset,
                                                              kwargs=self.constants_in_tree
                                                              )
        result_32 = np.float32(result)
        if np.all(np.isfinite(result_32)):
            return result_32
        else:
            raise NonFiniteError

    def fit_constants(self, call_node_id, dataset, mode):
        num_unfitted_constant = self.num_constants_in_complete_tree - self.constants_in_tree['num_fitted_constants']
        if num_unfitted_constant > 0:
            helper_obj = HelperObjectConstantFitting(
                node_to_evaluate=self.dict_of_nodes[call_node_id],
                node_id=call_node_id
            )
            p0 = [np.float32(1) for i in range(num_unfitted_constant)]
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    if mode == 'evaluate':
                        popt, pcov = curve_fit(
                            f=helper_obj.evaluate_subtree,
                            xdata=dataset.loc[:, dataset.columns != 'y'],
                            ydata=dataset.loc[:, 'y'],
                            p0=p0
                        )
                    else:
                        self.logger.error(f"Mode {mode} is not supported. Only"
                                          f"evaluate is supported")
                except OptimizeWarning as e:
                    popt = p0
                except ValueError as e:
                    popt = p0
                except NonFiniteError as e:
                    popt = p0
            self.set_constants(popt)

    def set_constants(self, popt):
        i = self.constants_in_tree['num_fitted_constants']
        for j in range(len(popt)):
            self.constants_in_tree[f"c_{i + j}"]['value'] = np.float32(popt[j])
            self.constants_in_tree['num_fitted_constants'] += 1

    def operators_data_range(self, variable):
        node_to_evaluate = self.dict_of_nodes[0]
        min_value, max_value, depends_on_variable = node_to_evaluate.math_class.operator_data_range(variable)
        return min_value, max_value

    def __str__(self):
        return self.rearrange_equation_prefix_notation(new_start_node_id=-1)[1]
