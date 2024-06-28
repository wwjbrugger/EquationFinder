from src.syntax_tree.syntax_tree import SyntaxTree
from src.game.game import Game, GameState
import typing
import numpy as np
from src.preprocess_data.pandas_preprocess import PandasPreprocess
from src.preprocess_data.gen_pandas_preprocess import GenPandasPreprocess
from copy import deepcopy
from src.utils.logging import get_log_obj
from src.constant_fitting.contant_fitting import refit_all_constants
import hashlib
import math
from src.game.rewards import Mse, ReMSe
from src.utils.error import NonFiniteError
from src.equation_classes.max_list import MaxList
import re


class FindEquationGame(Game):
    def __init__(self, grammar, args, train_test_or_val='train'):
        super().__init__(n_players=1)
        np.seterr(all='raise')
        self.logger = get_log_obj(args=args, name='Game')
        self.env_name = 'Find_Equation'
        self.grammar = grammar
        self.args = args
        self.max_list = MaxList(self.args)
        if self.args.equation_preprocess_class == 'PandasPreprocess':
            self.reader = PandasPreprocess(
                args=args,
                train_test_or_val=train_test_or_val,
                grammar=self.grammar
            )
        elif self.args.equation_preprocess_class == 'GenPandasPreprocess':
            self.reader = GenPandasPreprocess(
                args=args,
                train_test_or_val=train_test_or_val,
                grammar = self.grammar
            )
        else:
            raise NotImplementedError(f"Equation preprocess not defined: "
                                      f"{self.args.equation_preprocess_class}")

        self.iterator = self.reader.get_datasets()
        self.action_size = len(self.grammar._productions)
        self.dataset_columns = self.reader.dataset_columns

    def getInitialState(self) -> GameState:
        self.max_list = MaxList(self.args)
        batch_data = next(self.iterator)
        observations = {'data_frame': batch_data['data_frame'],
                        }
        if 'action_sequence' in batch_data:
            observations['action_sequence'] = batch_data['action_sequence']

        syntax_tree = SyntaxTree(
        args=self.args,
        grammar=self.grammar
        )
        syntax_tree.prefix_to_syntax_tree(prefix='S'.split())
        node_symbol, equation_str = syntax_tree.rearrange_equation_prefix_notation(
            new_start_node_id=-1
        )
        current_tree_representation_int = self.reader.map_tree_representation_to_int(
            symbol_list=equation_str.split())
        observations['current_tree_representation_str'] = equation_str
        observations['current_tree_representation_int'] = current_tree_representation_int
        observations['id_last_node'] = syntax_tree.nodes_to_expand[0]
        observations['last_symbol'] = \
            syntax_tree.dict_of_nodes[syntax_tree.nodes_to_expand[0]].node_symbol
        observations['true_equation'] = batch_data['infix_formula']
        observations['prefix_formula'] = batch_data['prefix_formula'] if  'prefix_formula' in batch_data else ''
        pattern = r'c_0_|___|__'
        matches = re.split(pattern, observations['true_equation'])
        observations['true_equation_hash'] = matches[0].strip()
        next_state = GameState(syntax_tree=syntax_tree, observation=observations)

        return next_state

    def getDimensions(self):
        pass

    def getActionSize(self) -> int:
        return self.action_size

    def getNextState(self, state: GameState, action: int, **kwargs) -> typing.Tuple[GameState, float]:
        next_tree = deepcopy(state.syntax_tree)
        next_tree.expand_node_with_action(
            node_id=state.syntax_tree.nodes_to_expand[0],
            action=action,
            build_syntax_tree_token_based=self.args.build_syntax_tree_token_based
        )
        node_symbol, equation = next_tree.rearrange_equation_prefix_notation(
            new_start_node_id=-1
        )
        next_observation = deepcopy(state.observation)
        next_observation['current_tree_representation_str'] = equation
        current_tree_representation_int = self.reader.map_tree_representation_to_int(
            symbol_list=equation.split())
        next_observation['current_tree_representation_int'] = current_tree_representation_int
        done = next_tree.complete or next_tree.max_depth_reached or \
               next_tree.max_constants_reached or next_tree.max_nodes_reached \
               or next_tree.invalid

        if done:
            next_observation['id_last_node'] = []
            next_observation['last_symbol'] = []
        else:
            next_observation['id_last_node'] = next_tree.nodes_to_expand[0]
            next_observation['last_symbol'] = \
                next_tree.dict_of_nodes[next_tree.nodes_to_expand[0]].node_symbol

        next_state = GameState(
            syntax_tree=next_tree,
            observation=next_observation,
            done=done,
            production_action=action,
            previous_state=state
        )

        reward = self.reward(state=next_state)
        next_state.reward = reward
        return next_state, reward

    def reward(self, state):
        dataset = state.observation['data_frame']
        syntax_tree = state.syntax_tree
        r = 0
        if syntax_tree.max_depth_reached:
            self.logger.debug('done max depth')
            r = self.args.minimum_reward
        elif syntax_tree.max_constants_reached:
            self.logger.debug('done max constant')
            r = self.args.minimum_reward
        elif syntax_tree.max_nodes_reached:
            self.logger.debug('done max number nodes')
            r = self.args.minimum_reward
        elif syntax_tree.invalid:
            self.logger.debug('Syntax tree is invalid ')
            r = self.args.minimum_reward
        elif syntax_tree.complete:
            try:
                complete_syntax_tree, initial_dataset = refit_all_constants(
                    finished_state=state,
                    args=self.args
                )
                syntax_tree.constants_in_tree = complete_syntax_tree.constants_in_tree
                state.complete_discovered_equation = syntax_tree.rearrange_equation_prefix_notation()[1]
                y_calc = syntax_tree.evaluate_subtree(
                    node_id=syntax_tree.start_node.node_id,
                    dataset=dataset,
                )
                y_true = dataset.loc[:, 'y'].to_numpy()
                error = Mse(y_pred=y_calc, y_true=y_true)  # ReMSe(y_pred=y_calc, y_true=y_true)
                # returns error in the range -1 to 1
                r = 1 + np.maximum(self.args.minimum_reward - 1, - error, dtype=np.float32)
                self.logger.debug(f"r = {r}  {syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1]} \n")

                if math.isfinite(r):
                    self.max_list.add(
                        state=state,
                        key=r
                    )
                else:
                    raise NonFiniteError

            except AssertionError:
                self.logger.debug(f"Equation can not be evaluated"
                                  f"the equation is: {syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1]} \n")
                r = float(self.args.minimum_reward)
            except FloatingPointError:
                self.logger.debug(f"In the calculation of the reward a FloatingPointError occur "
                                  f"the equation is: {syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=-1)[1]} \n"
                                  f"the dataset is: {dataset} ")
                r = float(self.args.minimum_reward)
            except RuntimeError as e:
                self.logger.debug(f"In the calculation of the reward a RuntimeError occur."
                                  f"The error is {e} "
                                  f"The previous state is used")
                r = float(self.args.minimum_reward)
            except NonFiniteError as e:
                r = float(self.args.minimum_reward)
        return r

    def getHash(self, state):
        data = np.ascontiguousarray(state.observation['data_frame'])
        hash1 = hashlib.md5(data).hexdigest()
        string_representation = (f"{state.syntax_tree.start_node.node_id}" \
                                 f"{state.observation['current_tree_representation_str']}_"
                                 f"{hash1}")
        if type(string_representation) != str:
            raise ValueError(f"Value to hash is of type {type(string_representation)} should be of type str")
        state.hash = string_representation
        return string_representation

    def getLegalMoves(self, state):
        possible_moves = state.syntax_tree.get_possible_moves(
            node_id=state.syntax_tree.nodes_to_expand[0]
        )
        temp = np.zeros(self.action_size)
        temp[possible_moves] = 1
        return temp

    def buildObservation(self):
        pass

    def getGameEnded(self):
        pass

    def getSymmetries(self):
        pass
