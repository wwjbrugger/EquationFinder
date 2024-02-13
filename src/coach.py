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
import timeit

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

        self.log_dir = f"{ROOT_DIR}/out/logs/AlphaZero/{run_name}"
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
        num_MCTS_sims = self.args.num_MCTS_sims
        self.logger.info(f"")
        self.logger.info(f"{mode}: equation for {state.observation['true_equation_hash']} is searched")
        start = time.time()
        while not state.done:
            # Compute the move probability vector and state value using MCTS for the current state of the environment.

            pi, v = mcts.run_mcts(
                state=state,
                num_mcts_sims=int(num_MCTS_sims),
                temperature=temp
            )
            # visualize_mcts(mcts)

            # Take a step in the environment and observe the transition and store necessary statistics.
            state.action = np.random.choice(len(pi), p=pi)
            next_state, r = game.getNextState(
                state=state,
                action=state.action,
            )
            complete_state.syntax_tree.expand_node_with_action(
                node_id=complete_state.syntax_tree.nodes_to_expand[0],
                action=state.action
            )

            history.capture(
                state=state,
                pi=pi,
                r=r,
                v=v
            )
            # Update state of control
            state = next_state
        end = time.time()


        # Cleanup environment and GameHistory
        if mcts.states_explored_till_perfect_fit > 0:
            wandb.log(
                {
                    f"num_states_to_perfect_fit_{mode}":
                        mcts.states_explored_till_perfect_fit,
                    f"{next_state.observation['true_equation_hash']}"
                    f"_num_states_to_perfect_fit_{mode}":
                        mcts.states_explored_till_perfect_fit,
                    f"time to perfect fit {next_state.observation['true_equation_hash']}":
                        end - start,
                    f"time to perfect fit":
                        end - start,
                    f"num simulation to perfect fit {next_state.observation['true_equation_hash']}":
                        mcts.num_simulation_till_perfect_fit,
                    f"num simulation to perfect fit":
                        mcts.num_simulation_till_perfect_fit

                }
            )
        else:
            wandb.log(
                {
                    f"equation_not_found_{next_state.observation['true_equation_hash']}_{mode}": 1,
                    f"equation_not_found_{mode}": 1,

                }
            )

        game.close(state)
        history.terminate(
            formula_started_from=formula_started_from,
            found_equation=complete_state.syntax_tree.rearrange_equation_infix_notation(-1)[1]
        )
        history.compute_returns(self.args, gamma=self.args.gamma, look_ahead=(
            self.args.n_steps if self.game.n_players == 1 else None
        ))
        return history

    def get_temperature(self, game):
        if game == self.game_test:
            temp = np.float32(0)
        else:
            try:
                temp = self.args.temp_0 * np.exp(
                    self.args.temperature_decay * np.float32(self.checkpoint.step.numpy())
                )
            except FloatingPointError:
                temp = 0
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
        t_end = time.time() + 60 * self.args.minutes_to_run
        self.metrics_train = {
            'mode': 'train',
            'rewards_mean': tf.keras.metrics.Mean(dtype=tf.float32),
            'done_rollout_ratio': tf.keras.metrics.Mean(dtype=tf.float32)
        }
        self.metrics_test = {
            'mode': 'test',
            'rewards_mean': tf.keras.metrics.Mean(dtype=tf.float32),
            'done_rollout_ratio': tf.keras.metrics.Mean(dtype=tf.float32)
        }
        self.loadTrainExamples(int(self.checkpoint.step))
        save_path = self.checkpoint_manager.save()
        while time.time() < t_end:
            self.logger.warning(f'------------------ITER'
                                f' {int(self.checkpoint.step)}----------------')
            # Self-play/ Gather training data.
            if not self.args.only_test:
                if self.args.run_mcts:
                    iteration_train_examples = self.gather_data(
                        metrics=self.metrics_train,
                        mcts=self.mcts,
                        game=self.game,
                        logger=self.logger,
                        num_selfplay_iterations=self.args.num_selfplay_iterations
                    )
                    self.trainExamplesHistory.append(iteration_train_examples)
                    wandb.log(
                        {f"average_reward_iteration_{self.metrics_train['mode']}":
                             self.metrics_train['rewards_mean'].result()}
                    )
                    wandb.log(
                        {f"average_done_rollout_ratio_iteration_{self.metrics_train['mode']}":
                             self.metrics_train['done_rollout_ratio'].result()}
                    )
                    self.saveTrainExamples(int(self.checkpoint.step))
                if self.args.prior_source in 'neural_net':
                    if self.checkpoint.step > self.args.cold_start_iterations:
                        self.update_network()
                    self.checkpoint.step.assign_add(1)
                    save_path = self.checkpoint_manager.save()
                    self.logger.debug(
                        f"Saved checkpoint for epoch {int(self.checkpoint.step)}: {save_path}"
                    )
                else:
                    self.checkpoint.step.assign_add(1)
                    save_path = self.checkpoint_manager.save()
            if self.args.test_network and \
                    self.checkpoint.step % self.args.test_every_n_steps == 0:
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

    def gather_data(self, metrics, mcts, game, logger, num_selfplay_iterations):
        iteration_train_examples = list()
        metrics['rewards_mean'].reset_state()
        metrics['done_rollout_ratio'].reset_state()
        minimal_reward_runs = 0
        for i in range(num_selfplay_iterations):
            mcts.clear_tree()
            result_episode = self.execute_one_game(
                game=game,
                mcts=mcts
            )
            if result_episode.observed_returns[0] == self.args.minimum_reward:
                minimal_reward_runs += 1
            iteration_train_examples.append(result_episode)
            self.log_best_list(game, logger)

            metrics['rewards_mean'].update_state(
                np.sum(result_episode.rewards, dtype=np.float32)
            )
            metrics['done_rollout_ratio'].update_state(
                mcts.visits_done_state / mcts.visits_roll_out
            )

        iteration_train_examples = self.augment_buffer(
            iteration_train_examples,
            metrics,
            minimal_reward_runs,
            num_selfplay_iterations
        )
        return iteration_train_examples

    def log_best_list(self, game, logger):
        logger.info(f"Best equations found:")
        for i in range(len(game.max_list.max_list_state)-1, 0,-1):
            logger.info(f"{i}: found equation: {game.max_list.max_list_state[i].complete_discovered_equation:<80}"
                        f" r={round(game.max_list.max_list_keys[i], 3)}"
                        )
            logger.info(game.max_list.max_list_state[i].syntax_tree.constants_in_tree)

    def augment_buffer(self, iteration_train_examples, metrics, minimal_reward_runs, num_selfplay_iterations):
        if metrics['mode'] == 'train':
            if self.args.balance_buffer:
                iteration_train_examples = self.balance_buffer(
                    iteration_train_examples=iteration_train_examples,
                    minimal_reward_runs=minimal_reward_runs,
                    num_selfplay_iterations=num_selfplay_iterations
                )
        return iteration_train_examples

    def balance_buffer(self, iteration_train_examples, minimal_reward_runs, num_selfplay_iterations):
        allowed_minimal_runs = int(num_selfplay_iterations * self.args.max_percent_of_minimal_reward_runs_in_buffer)
        if allowed_minimal_runs < minimal_reward_runs:
            balanced_iteration_train_examples = []
            allowed_minimal_runs_to_add = allowed_minimal_runs
            for example in iteration_train_examples:
                if example.observed_returns[0] > self.args.minimum_reward:
                    balanced_iteration_train_examples.append(example)
                elif allowed_minimal_runs_to_add > 0:
                    balanced_iteration_train_examples.append(example)
                    allowed_minimal_runs_to_add -= 1
                else:
                    pass
            return balanced_iteration_train_examples
        else:
            return iteration_train_examples

    def test_epoche(self, save_path):
        self.logger.warning(f'------------------ Test ----------------')
        self.checkpoint_test.restore(save_path)
        _ = self.gather_data(metrics=self.metrics_test,
                             mcts=self.mcts_test,
                             game=self.game_test,
                             logger=self.logger_test,
                             num_selfplay_iterations=self.args.num_selfplay_iterations_test
                             )
        wandb.log(
            {f"average_reward_{self.metrics_test['mode']}":
                 self.metrics_test['rewards_mean'].result()}
        )
        wandb.log(
            {f"average_done_rollout_ratio_{self.metrics_test['mode']}":
                 self.metrics_test['done_rollout_ratio'].result()}
        )

    def saveTrainExamples(self, iteration: int) -> None:
        """
        Store the current accumulated data to a compressed file using pickle. Note that for highly dimensional
        environments, that the stored files may be considerably large and that storing/ loading the data may
        introduce a significant bottleneck to the runtime of the algorithm.
        :param iteration: int Current iteration of the self-play. Used as indexing value for the data filename.
        """
        folder = ROOT_DIR / "saved_models" / self.args.data_path.name / \
                 'AlphaZero' / self.args.experiment_name / str(self.args.seed)

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
            folder = ROOT_DIR / "saved_models" / self.args.data_path.name / \
                     'AlphaZero' / self.args.experiment_name / str(self.args.seed)
            buffer_number = highest_number_in_files(path=folder, stem='buffer_')
            filename = folder / f"buffer_{buffer_number}.examples"

            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    self.logger.info(f"Replay buffer {buffer_number}  found. Read it.")
                    self.trainExamplesHistory = Unpickler(f).load()
            else:
                self.logger.info(f"No replay buffer found. Use empty one.")
