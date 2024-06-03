"""
Define the base self-play/ data gathering class. This class should work with any MCTS-based neural network learning
algorithm like AlphaZero or MuZero. Self-play, model-fitting, and pitting is performed sequentially on a single-thread
in this default implementation.

Notes:
 - Code adapted from https://github.com/suragnair/alpha-zero-general
 - Base implementation done.
 - Base implementation sufficiently abstracted to accommodate both AlphaZero and MuZero.
 - Documentation 15/11/2020
"""
import copy
import logging
import math
import os
import sys
import typing
from pickle import Pickler, Unpickler, HIGHEST_PROTOCOL
from collections import deque
from abc import ABC

import numpy as np
from tqdm import trange

from src.game.game_history import GameHistory, \
    sample_batch
import time
from datetime import datetime
import tensorflow as tf
import wandb
from src.utils.logging import get_log_obj
from src.utils.files import highest_number_in_files
from definitions import ROOT_DIR
import random
from src.preprocess_data.equation_preprocess_dummy import (
    equation_to_action_sequence, get_dict_token_to_action)

class Coach(ABC):
    """
    This class controls the self-play and learning loop.
    """

    def __init__(self, game, game_test, rule_predictor, rule_predictor_test, args, search_engine, run_name, checkpoint_train, checkpoint_manager, checkpoint_test=None) -> None:
        """
        Initialize the self-play class with an environment, an agent to train, requisite hyperparameters, a MCTS search
        engine, and an agent-interface.
        :param rule_predictor_test:
        :param game_test:
        :param run_name:
        :param game: Game Implementation of Game class for environment logic.
        :param rule_predictor: Some implementation of a neural network class to be trained.
        :param args Data structure containing parameters for self-play.
        :param search_engine: Class containing the logic for performing MCTS using the neural_net.
        """

        self.game = game
        self.game_test = game_test
        self.args = args

        # Initialize replay buffer and helper variable
        self.trainExamplesHistory = deque(
            maxlen=self.args.selfplay_buffer_window)

        # Initialize network and search engine
        self.rule_predictor = rule_predictor
        self.rule_predictor_test = rule_predictor_test
        self.mcts = search_engine(self.game,
                                  self.rule_predictor,
                                  self.args
                                  )
        self.mcts_test = search_engine(self.game_test,
                                       self.rule_predictor_test,
                                       self.args
                                       )

        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.log_dir = f"{ROOT_DIR}/out/logs/{run_name}"
        self.file_writer = tf.summary.create_file_writer(
            self.log_dir + "/metrics")
        self.file_writer.set_as_default()
        self.checkpoint = checkpoint_train
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_test = checkpoint_test
        self.logger = get_log_obj(args=args, name='coach')
        self.logger_test = get_log_obj(args=args, name='coach_test')

    @staticmethod
    def getCheckpointFile(iteration: int) -> str:
        """ Helper function to format model checkpoint filenames """
        return f'checkpoint_{iteration}.pth.tar'

    def sampleBatch(self, histories: typing.List[GameHistory]) -> typing.List:
        """
          Sample a batch of data from the current replay buffer (with or without prioritization).
        Construct a batch of data-targets for gradient optimization of the AlphaZero neural network.

        The procedure samples a list of game and inside-game coordinates of length 'batch_size'. This is done either
        uniformly or with prioritized sampling. Using this list of coordinates, we sample the according games, and
        the according points of times within the game to generate neural network inputs, targets, and sample weights.

        The targets for the neural network consist of MCTS move probability vectors and TD/ Monte-Carlo returns.

        :param histories: List of GameHistory objects. Contains all game-trajectories in the replay-buffer.
        :return: List of training examples: (observations, (move-probabilities, TD/ MC-returns), sample_weights)
        """
        # Generate coordinates within the replay buffer to sample from. Also generate the loss scale of said samples.
        sample_coordinates, sample_weight = sample_batch(
            list_of_histories=histories,
            n=self.args.batch_size_training,
            prioritize=self.args.prioritize,
            alpha=self.args.prioritize_alpha,
            beta=self.args.prioritize_beta
        )

        # Collect training examples for AlphaZero: (o_t, (pi_t, v_t), w_t)
        examples = [
            {
                'observation': histories[h_i].stackObservations(
                    length=1, t=i),
                'probabilities_actor': histories[h_i].probabilities[i],
                'observed_return': histories[h_i].observed_returns[i],
                'loss_scale': loss_scale,
                'found_equation': histories[h_i].found_equation,
            }
            for (h_i, i), loss_scale in zip(sample_coordinates, sample_weight)
        ]
        return examples

    def execute_one_game(self, game, mcts) -> GameHistory:
        """
        Perform one episode of self-play for gathering data to train neural networks on.

        The implementation details of the neural networks/ agents, temperature schedule, data storage
        is kept highly transparent on this side of the algorithm. Hence for implementation details
        see the specific implementations of the function calls.

        At every step we record a snapshot of the state into a GameHistory object, this includes the observation,
        MCTS search statistics, performed action, and observed rewards. After the end of the episode, we close the
        GameHistory object and compute internal target values.

        :return: GameHistory Data structure containing all observed states and statistics required for network training.
        """
        history = GameHistory()
        state = game.getInitialState()  # Always from perspective of player 1 for boardgames.
        complete_state = copy.deepcopy(state)
        formula_started_from = state.observation['current_tree_representation_str']
        # Update MCTS visit count temperature according to an episode or weight update schedule.
        temp = self.get_temperature(game)
        if game == self.game_test:
            mode = 'test'
        else:
            mode = 'train'
        wandb.log({f"temperature_{mode}": temp})

        self.logger.info(f"")
        self.logger.info(f"{mode}: equation for {state.observation['true_equation_hash']} is searched")

        # Compute the move probability vector and state value using MCTS for the current state of the environment.

        pi, v = self.get_mcts_action(mcts, mode, state, temp)

        # Take a step in the environment and observe the transition and store necessary statistics.
        if mode == 'test':
            state.action = np.argmax(pi)
        state.action = np.random.choice(len(pi), p=pi)
        next_state, r = game.getNextState(
            state=state,
            action=state.action,
        )
        complete_state.syntax_tree.expand_node_with_action(
            node_id=complete_state.syntax_tree.nodes_to_expand[0],
            action=state.action,
            build_syntax_tree_token_based=self.args.build_syntax_tree_token_based
        )

        wandb.log(
            {
                "successful": True if mcts.states_explored_till_0_999 >= 0 else False
            }
        )
        self.log_mcts_results(self.game,mcts )

    def log_mcts_results(self, game, mcts):
        # Cleanup environment and GameHistory
        self.logger.info(f"Initial guess of NN: ")
        initial_hash = list(mcts.Ps.keys())[0]
        self.logger.info(np.around(mcts.Ps[initial_hash], 2))
        for i in np.where(mcts.valid_moves_for_s[initial_hash])[0]:
                self.logger.info(f"     {str(game.grammar._productions[i]._rhs) :<120}|"
                                 f" Ps: {round(mcts.Ps[initial_hash][i], 2):<10.2f}|"
                                 )

        return

    def get_mcts_action(self, mcts, mode, state, temp):
        pi, v = mcts.run_mcts(
            state=state,
            temperature=temp
        )
        # Take a step in the environment and observe the transition and store necessary statistics.
        if mode == 'test':
            state.action = np.argmax(pi)
        state.action = np.random.choice(len(pi), p=pi)
        return pi, v

    def get_temperature(self, game):
        try:
            temp = self.args.temp_0 * np.exp(
                self.args.temperature_decay * np.float32(self.checkpoint.step.numpy())
            )
        except FloatingPointError:
            temp = self.args.temp_0
        return temp

    def learn(self) -> None:
        """
        Control the data gathering and weight optimization loop. Perform 'num_selfplay_iterations' iterations
        of self-play to gather data, each of 'num_episodes' episodes. After every self-play iteration, train the
        neural network with the accumulated data. If specified, the previous neural network weights are evaluated
        against the newly fitted neural network weights, the newly fitted weights are then accepted based on some
        specified win/ lose ratio. Neural network weights and the replay buffer are stored after every iteration.
        Note that for highly granular vision based environments, that the replay buffer may grow to large sizes.
        """
        self.metrics_test = {
            'mode': 'test',
            'rewards_mean': tf.keras.metrics.Mean(dtype=tf.float32),
            'done_rollout_ratio': tf.keras.metrics.Mean(dtype=tf.float32)
        }
        save_path = self.checkpoint_manager.save()

        self.logger.warning(f'------------------ITER'
                            f' {int(self.checkpoint.step)}----------------')
        # Self-play/ Gather training data.

        self.test_epoche(save_path=save_path)

    def update_network(self):
        # Flatten examples over self-play episodes and sample a training batch.
        complete_history = GameHistory.flatten(self.trainExamplesHistory)
        logging.warning(f"Number of samples in Replay buffer {len(complete_history)}")
        # Backpropagation
        train_pi_loss = 0
        train_v_loss = 0
        for _ in trange(self.args.num_gradient_steps,
                        desc="Backpropagation",
                        file=sys.stdout):
            batch = self.sampleBatch(complete_history)
            pi_batch_loss, v_batch_loss, contrastive_loss = \
                self.rule_predictor.train(batch)
            train_pi_loss += pi_batch_loss
            train_v_loss += v_batch_loss
            wandb.log({f"Train pi loss": pi_batch_loss})
            wandb.log({f"Train v loss": v_batch_loss})
            if self.args.contrastive_loss:
                wandb.log({f"Contrastive loss": contrastive_loss})

    def gather_data(self, mcts, game, logger):
        minimal_reward_runs = 0
        mcts.clear_tree()
        result_episode = self.execute_one_game(
            game=game,
            mcts=mcts
        )

        self.log_best_list(game, logger)

        return

    def log_best_list(self, game, logger):
        logger.info(f"Best equations found:")
        for i in range(len(game.max_list.max_list_state) -1  , - 1, -1):
            logger.info(f"{i}: found equation: {game.max_list.max_list_state[i].complete_discovered_equation:<80}"
                        f" r={round(game.max_list.max_list_keys[i], 3)}"
                        )
            logger.info(game.max_list.max_list_state[i].syntax_tree.constants_in_tree)

    def augment_buffer(self, iteration_examples, metrics, minimal_reward_runs, num_selfplay_iterations):
        if metrics['mode'] == 'train':
            if self.args.balance_buffer:
                iteration_examples = self.balance_buffer(
                    iteration_examples=iteration_examples,
                    minimal_reward_runs=minimal_reward_runs,
                    num_selfplay_iterations=num_selfplay_iterations
                )
        return iteration_examples

    def balance_buffer(self, iteration_examples, minimal_reward_runs, num_selfplay_iterations):
        allowed_minimal_runs = int(num_selfplay_iterations * self.args.max_percent_of_minimal_reward_runs_in_buffer)
        if allowed_minimal_runs < minimal_reward_runs:
            balanced_iteration_examples = []
            allowed_minimal_runs_to_add = allowed_minimal_runs
            for example in iteration_examples:
                if example.observed_returns[0] > self.args.minimum_reward:
                    balanced_iteration_examples.append(example)
                elif allowed_minimal_runs_to_add > 0:
                    balanced_iteration_examples.append(example)
                    allowed_minimal_runs_to_add -= 1
                else:
                    pass
            return balanced_iteration_examples
        else:
            return iteration_examples

    def test_epoche(self, save_path):
        self.logger.warning(f'------------------ Test ----------------')
        self.checkpoint_test.restore(save_path)
        sim = []
        states = []
        unsuccessful_runs = 0
        start_time = time.time()
        for i in range(self.args.num_selfplay_iterations_test):
            self.logger_test.info(f"iteration: {i}")
            self.gather_data(mcts=self.mcts_test,
                             game=self.game_test,
                             logger=self.logger_test,
                             )
            sim_0_999 = self.mcts_test.num_simulation_till_0_999
            states_0_999 = self.mcts_test.states_explored_till_0_999
            if sim_0_999 > 0:
                sim.append(sim_0_999)
                states.append(states_0_999)
                productions=[]
                for node in self.game_test.max_list.max_list_state[-1].syntax_tree.dict_of_nodes.values():
                    if len(node.selected_production) > 0:
                        productions.append(  str(node.selected_production))
                print(f"num productions: {len(productions)} : {productions} ")

            else:
                unsuccessful_runs += 1

        self.logger_test.info(f"sim:{sim}")
        self.logger_test.info(f"states:{states}")
        self.logger_test.info(f"avg. sim: {np.mean(sim)} \n avg. states: {np.mean(states)}")
        wandb.log({
            'equation_found': self.game_test.max_list.max_list_state[-1].syntax_tree.__str__(),
            'productions': productions,
            'run_time': time.time() - start_time,
            'sim': sim,
            'states': states,
            "avg. sim": np.mean(sim),
            "avg. states": np.mean(states),
            'unsuccessful_runs': unsuccessful_runs
        })

    def saveTrainExamples(self, iteration: int) -> None:
        """
        Store the current accumulated data to a compressed file using pickle. Note that for highly dimensional
        environments, that the stored files may be considerably large and that storing/ loading the data may
        introduce a significant bottleneck to the runtime of the algorithm.
        :param iteration: int Current iteration of the self-play. Used as indexing value for the data filename.
        """
        folder = ROOT_DIR / 'saved_models' / self.args.data_path / \
                          str(self.args.experiment_name) / str(self.args.seed)

        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = folder / f"buffer_{iteration}.examples"
        with open(filename, "wb+") as f:
            Pickler(f, protocol=HIGHEST_PROTOCOL).dump(
                self.trainExamplesHistory)

        # Don't hog up storage space and clean up old (never to be used again) data.
        old_checkpoint = folder / f"buffer_{iteration - 1}.examples"
        if os.path.isfile(old_checkpoint):
            os.remove(old_checkpoint)

    def loadTrainExamples(self, iteration: int) -> None:
        """
        Load in a previously generated replay buffer from the path specified in the .json arguments.
        """
        if len(self.args.replay_buffer_path) >= 1:
            if os.path.isfile(self.args.replay_buffer_path):
                with open(self.args.replay_buffer_path, "rb") as f:
                    self.logger.info(f"Replay buffer {self.args.replay_buffer_path}  found. Read it.")
                    self.trainExamplesHistory = Unpickler(f).load()
            else:
                self.logger.info(f"No replay buffer found. Use empty one.")
        else:
            folder = ROOT_DIR / 'saved_models' / self.args.data_path / \
                          str(self.args.experiment_name) / str(self.args.seed)
            buffer_number = highest_number_in_files(path=folder, stem='buffer_')
            filename = folder / f"buffer_{buffer_number}.examples"

            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    self.logger.info(f"Replay buffer {buffer_number}  found. Read it.")
                    self.trainExamplesHistory = Unpickler(f).load()
            else:
                self.logger.info(f"No replay buffer found. Use empty one.")
