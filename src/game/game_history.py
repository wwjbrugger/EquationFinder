"""
Defines the functionality for prioritized sampling, the replay-buffer, min-max normalization, and parameter scheduling.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import typing

import numpy as np
from src.game.game import GameState


@dataclass
class GameHistory:
    """
    Data container for keeping track of game trajectories.
    """
    observations: list = field(default_factory=list)        # o_t: State Observations
    players: list = field(default_factory=list)             # p_t: Current player
    probabilities: list = field(default_factory=list)       # pi_t: Probability vector of MCTS for the action
    MCTS_value_estimation: list = field(default_factory=list)      # v_t: MCTS value estimation
    rewards: list = field(default_factory=list)             # u_t+1: Observed reward after performing a_t+1
    actions: list = field(default_factory=list)             # a_t+1: Action leading to transition s_t -> s_t+1
    observed_returns: list = field(default_factory=list)    # z_t: Training targets for the value function
    terminated: bool = False                                # Whether the environment has terminated

    def __len__(self) -> int:
        """Get length of current stored trajectory"""
        return len(self.observations)

    def capture(self, state: GameState, pi: np.ndarray, r: float, v: float) -> None:
        """Take a snapshot of the current state of the environment and the search results"""
        self.observations.append(state.observation)
        self.actions.append(state.action)
        self.probabilities.append(pi)
        self.rewards.append(r)
        self.MCTS_value_estimation.append(v)

    def terminate(self, formula_started_from='', found_equation='') -> None:
        """Take a snapshot of the terminal state of the environment"""
        # self.probabilities.append(np.zeros_like(self.probabilities[-1]))
        # self.rewards.append(0)         # Reward past u_T
        # self.MCTS_value_estimation.append(0)
        self.formula_started_from = formula_started_from# Bootstrap: Future possible reward = 0
        self.found_equation = found_equation

        self.terminated = True

    def refresh(self) -> None:
        """Clear all statistics within the class"""
        all([x.clear() for x in vars(self).values() if type(x) == list])
        self.terminated = False

    def compute_returns(self, args, gamma: float = 1, look_ahead: typing.Optional[int] = None) -> None:
        """Computes the n-step returns assuming that the last recorded snapshot was a terminal state
        :param args:
        """
        self.observed_returns = list()
        horizon = len(self.rewards)
        for t in range(len(self.rewards)):
            discounted_rewards = [np.power(gamma, k - t) * self.rewards[k] for k in range(t, horizon)]
            observed_return = sum(discounted_rewards) #+ bootstrap
            if args.average_policy_if_wrong and observed_return < args.maximum_reward - 0.99 :
                self.probabilities[t][self.probabilities[t] > 0] = 1/np.count_nonzero(self.probabilities[t])
            self.observed_returns.append(observed_return)
        return

    def stackObservations(self, length: int, current_observation: typing.Optional[np.ndarray] = None,
                          t: typing.Optional[int] = None) -> np.ndarray:  # TODO: rework function.
        """Stack the most recent 'length' elements from the observation list along the end of the observation axis"""
        if length <= 1:
            if current_observation is not None:
                return current_observation
            elif t is not None:
                return self.observations[np.min([t, len(self) - 1])]
            else:
                return self.observations[-1]

        if t is None:
            # If current observation is also None, then t needs to both index and slice self.observations:
            # for len(self) indexing will throw an out of bounds error when current_observation is None.
            # for len(self) - 1, if current_observation is NOT None, then the trajectory wil omit a step.
            # Proof: t = len(self) - 1 --> list[:t] in {i, ..., t-1}.
            t = len(self) - (1 if current_observation is None else 0)

        if current_observation is None:
            current_observation = self.observations[t]

        # Get a trajectory of states of 'length' most recent observations until time-point t.
        # Trajectories sampled beyond the end of the game are simply repeats of the terminal observation.
        if t > len(self):
            terminal_repeat = [current_observation] * (t - len(self))
            trajectory = self.observations[:t][-(length - len(terminal_repeat)):] + terminal_repeat
        else:
            trajectory = self.observations[:t][-(length - 1):] + [current_observation]

        if len(trajectory) < length:
            prefix = [np.zeros_like(current_observation) for _ in range(length - len(trajectory))]
            trajectory = prefix + trajectory

        return np.concatenate(trajectory, axis=-1)  # Concatenate along channel dimension.

    @staticmethod
    def print_statistics(histories: typing.List[typing.List[GameHistory]]) -> None:
        """ Print generic statistics over a nested list of GameHistories (the entire Replay-Buffer). """
        flat = GameHistory.flatten(histories)

        n_self_play_iterations = len(histories)
        n_episodes = len(flat)
        n_samples = sum([len(x) for x in flat])

        print("=== Replay Buffer Statistics ===")
        print(f"Replay buffer filled with data from {n_self_play_iterations} self play iterations")
        for history in flat:
            print(f"searched equation: {history.searched_equation},  started from : {history.formula_started_from},  found equation: {history.found_equation}")
        print(f"In total {n_episodes} episodes have been played amounting to {n_samples} data samples")


    @staticmethod
    def flatten(nested_histories: typing.List[typing.List[GameHistory]]) -> typing.List[GameHistory]:
        """ Flatten doubly nested list to a normal list of objects. """
        return [subitem for item in nested_histories for subitem in item]


def sample_batch(list_of_histories: typing.List[GameHistory], n: int, prioritize: bool = False, alpha: float = 1.0,
                 beta: float = 1.0) -> typing.Tuple[typing.List[typing.Tuple[int, int]], typing.List[float]]:
    """
    Generate a sample specification from the list of GameHistory object using uniform or prioritized sampling.
    Along with the generated indices, for each sample/ index a scalar is returned for the loss function during
    backpropagation. For uniform sampling this is simply w_i = 1 / N (Mean) for prioritized sampling this is
    adjusted to

        w_i = (1/ (N * P_i))^beta,

    where P_i is the priority of sample/ index i, defined as

        P_i = (p_i)^alpha / sum_k (p_k)^alpha, with p_i = |v_i - z_i|,

    and v_i being the MCTS search result and z_i being the observed n-step return.
    The w_i is the Importance Sampling ratio and accounts for some of the sampling bias.

    WARNING: Sampling with replacement is performed if the given batch-size exceeds the replay-buffer size.

    :param list_of_histories: List of GameHistory objects to sample indices from.
    :param n: int Number of samples to generate == batch_size.
    :param prioritize: bool Whether to use prioritized sampling
    :param alpha: float Exponentiation factor for computing priorities, high = uniform, low = greedy
    :param beta: float Exponentiation factor for the Importance Sampling ratio.
    :return: List of tuples indicating a sample, the first index in the tuple specifies which GameHistory object
             within list_of_histories is chosen and the second index specifies the time point within that GameHistory.
             List of scalars containing either the Importance Sampling ratio or 1 / N to scale the network loss with.
    """
    lengths = list(map(len, list_of_histories))   # Map the trajectory length of each Game

    sampling_probability = None                   # None defaults to uniform in np.random.choice
    sample_weight = np.ones(np.sum(lengths))      # 1 / N. Uniform weight update strength over batch.

    if prioritize or alpha == 0:
        errors = np.array([np.abs(h.MCTS_value_estimation[i] - h.observed_returns[i])
                           for h in list_of_histories for i in range(len(h))])

        mass = np.power(errors, alpha)
        sampling_probability = mass / np.sum(mass)

        # Adjust weight update strength proportionally to IS-ratio to account for sampling bias.
        sample_weight = np.power(n * sampling_probability, beta)

    # Sample with prioritized / uniform probabilities sample indices over the flattened list of GameHistory objects.
    flat_indices = np.random.choice(a=np.sum(lengths), size=n, replace=(n > np.sum(lengths)), p=sampling_probability)

    # Map the flat indices to the correct histories and history indices.
    history_index_borders = np.cumsum(lengths)
    history_indices = [(np.sum(i >= history_index_borders), i) for i in flat_indices]

    # Of the form [(history_i, t), ...] \equiv history_it
    sample_coordinates = [(h_i, i - np.r_[0, history_index_borders][h_i]) for h_i, i in history_indices]
    # Extract the corresponding IS loss scalars for each sample (or simply N x 1 / N if non-prioritized)
    sample_weights = sample_weight[flat_indices]

    return sample_coordinates, sample_weights
