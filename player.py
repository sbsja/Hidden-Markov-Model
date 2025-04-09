#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import math
import time
import sys

class HiddenMarkovModel:
    def __init__(self, n_states, n_emissions):
        """
        Creates a new Hidden Markov Model with n_states states and n_observations possible observations.
        :param n_states: number of states
        :param n_observations: number of possible observations
        """
        self.n_states = n_states
        self.n_observations = n_emissions
        self.PI = generate_row_stochastic_matrix(1, n_states)
        self.A = generate_row_stochastic_matrix(n_states, n_states)
        self.B = generate_row_stochastic_matrix(n_states, n_emissions)

    def __str__(self):
        return "PI: {}\nA: {}\nB: {}".format(self.PI, self.A, self.B)

    def set_A(self, A):
        self.A = A

    def set_B(self, B):
        self.B = B

    def set_PI(self, PI):
        self.PI = PI

    def update_model(self, observations):
        """
        Updates the model's parameters using the Baum-Welch algorithm.
        :param observations: a list of observations
        """
        start_time = time.time()
        self.A, self.B, self.PI = baum_welch(
            self.A, self.B, self.PI, observations, max_iter=10)
        # print("     Time to update model: {:.3f}s".format(time.time() - start_time))

    def get_most_probable_sequence(self, observations):
        """
        Returns the most probable sequence of states given a list of observations.
        :param observations: a list of observations
        :return: a list of states
        """
        return viterbi(self.A, self.B, self.PI, observations)

    def get_probability(self, observations):
        """
        Returns the probability of a sequence of observations.
        :param observations: a list of observations
        :return: a float
        """
        return forward_algorithm(self.A, self.B, self.PI, observations)

    def find_most_probable_state(self, observations):
        """
        Returns the most probable state given a list of observations.
        :param observations: a list of observations
        :return: a list of states
        """
        alpha, _ = alpha_pass(self.A, self.B, self.PI, observations)
        return max(range(len(alpha[-1])), key=lambda i: alpha[-1][i])


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        Initialize the HMM models and data storage.
        """
        self.species_models = [HiddenMarkovModel(1, N_EMISSIONS) for _ in range(N_SPECIES)]  # One model per species
        self.observations = [[] for _ in range(N_FISH)]  # Observation sequences for each fish
        self.guessed_fishes = [False] * N_FISH  # Track fishes that have been guessed

    def guess(self, step, current_observations):
        """
        Process observations and make a guess about the fish type.
        :param step: Current step number in the game.
        :param current_observations: List of current observations for all fishes.
        :return: None or a tuple (fish_index, guessed_species).
        """
        # Update observation sequences for fishes not yet guessed
        for fish_index in range(N_FISH):
            if not self.guessed_fishes[fish_index]:
                self.observations[fish_index].append(current_observations[fish_index])

        # Start guessing after a threshold step
        if step < 110:
            return None

        # Select a random fish that hasn't been guessed
        fish_index = random.choice([i for i in range(N_FISH) if not self.guessed_fishes[i]])
        max_probability = 0
        predicted_species = random.randint(0, N_SPECIES - 1)  # Default to a random guess

        # Evaluate the probability of the fish belonging to each species
        for species_index, model in enumerate(self.species_models):
            probability = model.get_probability(self.observations[fish_index])
            if probability > max_probability:
                max_probability = probability
                predicted_species = species_index

        return fish_index, predicted_species

    def reveal(self, correct, fish_index, true_species):
        """
        Handle the result of a guess and update the corresponding model.
        :param correct: Whether the guess was correct.
        :param fish_index: Index of the fish that was guessed.
        :param true_species: Actual species of the fish.
        """
        self.guessed_fishes[fish_index] = True  # Mark the fish as guessed
        if not correct:
            self.species_models[true_species].update_model(self.observations[fish_index])

epsilon = sys.float_info.epsilon  # avoid division by zero

def generate_row_stochastic_matrix(n, m):
    """
    Generates a n x m matrix with random values in the range [0, 1]
    and normalizes each row to sum up to 1.
    :param n: number of rows
    :param m: number of columns
    :return: a n x m matrix
    """
    matrix = [[1 / m + random.random() / 1000 for _ in range(m)]
              for _ in range(n)]
    for row in matrix:
        row_sum = sum(row)
        for i in range(len(row)):
            row[i] /= row_sum
    return matrix


def multiply(A, B):
    return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]


def print_list(L):
    for l in L:
        print("{:.3f}".format(l), end=" ")
        # print(l, end=" ")


def print_matrix(M):
    for row in M:
        print_list(row)
        print()
    print()

def emissions_probability(A, B, pi):
    return multiply(multiply(pi, A), B)


def forward_algorithm(A, B, pi, obs):
    alpha = [[pi[0][i] * B[i][obs[0]] for i in range(len(pi[0]))]]
    for t in range(1, len(obs)):
        alpha.append([sum([alpha[t - 1][j] * A[j][i]
                           for j in range(len(A))]) * B[i][obs[t]] for i in range(len(A))])
    return sum(alpha[-1])


def viterbi(A, B, pi, emissions):
    """
    Implements the Viterbi Algorithm to find the most likely sequence of states.
    :param A: Transition matrix
    :param B: Emission matrix
    :param pi: Initial state probability distribution
    :param emissions: List of observed emissions
    :return: List of the most probable sequence of states
    """
    T = len(emissions) 
    N = len(pi)
    
    delta = [[0] * N for i in range(T)] # Tracks most likely probabilities each time step
    backpointer = [[0] * N for i in range(T)] # Stores the indices of the states that contributed to those paths
    
    # Initialization for t = 0
    for i in range(N):
        delta[0][i] = pi[i] * B[i][emissions[0]]
        backpointer[0][i] = 0 # No predecessor

    # Recursion for t = 1 to T-1
    for t in range(1, T):  
        for current_state in range(N):  
            max_probability = 0
            best_previous_state = 0
            
            for previous_state in range(N):
                transition_probability = delta[t-1][previous_state] * A[previous_state][current_state]
                
                if transition_probability > max_probability:
                    max_probability = transition_probability
                    best_previous_state = previous_state
            
            delta[t][current_state] = max_probability * B[current_state][emissions[t]]
            backpointer[t][current_state] = best_previous_state

    # Find the state with the highest probability at the final time step
    max_final_prob = 0
    final_state = 0
    for state in range(N):
        if delta[T-1][state] > max_final_prob:
            max_final_prob = delta[T-1][state]
            final_state = state

    
    # Backtracking to find the most probable state sequence
    most_likely_sequence = [0] * T
    most_likely_sequence[-1] = final_state
    for t in range(T-2, -1, -1):
        most_likely_sequence[t] = backpointer[t+1][most_likely_sequence[t+1]]
    
    return most_likely_sequence


def alpha_pass(A, B, pi, obs):
    alpha = []
    scalers = []  # introduced it to avoid underflow

    alpha.append([pi[0][i] * B[i][obs[0]] for i in range(len(pi[0]))])
    scalers.append(1 / (sum(alpha[0]) + epsilon))

    alpha[0] = [alpha_0_i * scalers[0] for alpha_0_i in alpha[0]]

    for t in range(1, len(obs)):
        alpha.append([sum([alpha[t - 1][j] * A[j][i]
                           for j in range(len(A))]) * B[i][obs[t]] for i in range(len(A))])
        scalers.append(1 / (sum(alpha[t]) + epsilon))
        alpha[t] = [alpha_t_i * scalers[t] for alpha_t_i in alpha[t]]

    return alpha, scalers


def beta_pass(A, B, obs, scalers):
    beta = [[scalers[-1] for _ in range(len(A))]]
    for t in range(len(obs) - 2, -1, -1):
        beta.insert(0, [sum([beta[0][j] * A[i][j] * B[j][obs[t + 1]]
                             for j in range(len(A))]) for i in range(len(A))])
        beta[0] = [beta_0_i * scalers[t] for beta_0_i in beta[0]]
    return beta


def get_gammas(A, B, alpha, beta, obs):
    gamma = []
    di_gamma = []

    for t in range(len(obs) - 1):
        di_gamma.append([[alpha[t][i] * A[i][j] * B[j][obs[t + 1]]
                          * beta[t + 1][j] for j in range(len(A))] for i in range(len(A))])
        gamma.append([sum(di_gamma[t][i]) for i in range(len(A))])
    gamma.append(alpha[-1])
    return gamma, di_gamma


def re_estimate(A, B, pi, obs):
    alpha, scalers = alpha_pass(A, B, pi, obs)
    beta = beta_pass(A, B, obs, scalers)
    gamma, di_gamma = get_gammas(A, B, alpha, beta, obs)

    new_pi = [[gamma[0][i] for i in range(len(A))]]
    new_A = [[sum([di_gamma[t][i][j] for t in range(len(obs) - 1)])
              / (sum([gamma[t][i] for t in range(len(obs) - 1)]) + epsilon) for j in range(len(A))] for i in
             range(len(A))]
    new_B = [[sum([gamma[t][j] for t in range(len(obs)) if obs[t] == k])
              / (sum([gamma[t][j] for t in range(len(obs))]) + epsilon) for k in range(len(B[0]))] for j in
             range(len(A))]

    return new_A, new_B, new_pi, scalers


def compute_log_likelihood(scalers):
    return -sum([math.log(s) for s in scalers])


def baum_welch(A, B, pi, obs, max_iter=5):
    new_A, new_B, new_pi, scalers = re_estimate(A, B, pi, obs)
    previous_log_likelihood = float("-inf")

    iterations = 0
    while True:
        A, B, pi = new_A, new_B, new_pi
        current_log_likelihood = compute_log_likelihood(scalers)
        if previous_log_likelihood > current_log_likelihood or iterations >= max_iter:
            break
        iterations += 1
        previous_log_likelihood = current_log_likelihood
        new_A, new_B, new_pi, scalers = re_estimate(A, B, pi, obs)

    return new_A, new_B, new_pi
