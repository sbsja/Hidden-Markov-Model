import math
import time
import random

# Utility functions
def normalize_row(row, threshold=1e-8):
    row = [x if x > threshold else 0 for x in row]
    row_sum = sum(row)
    return [x / row_sum for x in row] if row_sum > 0 else row

def normalize_matrix(matrix, threshold=1e-8):
    return [normalize_row(row, threshold) for row in matrix]

# Forward and backward passes
def alpha_pass(A, B, pi, emissions):
    alpha = []
    scalers = []
    alpha.append([pi[i] * B[i][emissions[0]] for i in range(len(pi))])
    scalers.append(1 / sum(alpha[0]))
    alpha[0] = [alpha_t_i * scalers[0] for alpha_t_i in alpha[0]]
    for t in range(1, len(emissions)):
        alpha.append([
            sum(alpha[t-1][j] * A[j][i] for j in range(len(A))) * B[i][emissions[t]]
            for i in range(len(A))
        ])
        scalers.append(1 / sum(alpha[t]))
        alpha[t] = [alpha_t_i * scalers[t] for alpha_t_i in alpha[t]]
    return alpha, scalers

def beta_pass(A, B, emissions, scalers):
    beta = [[scalers[-1] for _ in range(len(A))]]
    for t in range(len(emissions) - 2, -1, -1):
        beta.insert(0, [
            sum(A[i][j] * B[j][emissions[t + 1]] * beta[0][j] for j in range(len(A)))
            * scalers[t]
            for i in range(len(A))
        ])
    return beta

# Re-estimation of parameters
def re_estimate(A, B, pi, emissions, alpha, beta, threshold=1e-8):
    """
    Re-estimate A, B, and pi using gamma and di-gamma values.
    """
    N = len(A)
    M = len(B[0])
    T = len(emissions)

    gamma = []
    di_gamma = []

    for t in range(T - 1):
        di_gamma.append([
            [
                alpha[t][i] * A[i][j] * B[j][emissions[t + 1]] * beta[t + 1][j]
                for j in range(N)
            ]
            for i in range(N)
        ])
        gamma.append([sum(di_gamma[t][i]) for i in range(N)])
    gamma.append(alpha[-1])

    # Re-estimate pi
    new_pi = gamma[0]

    # Re-estimate A
    new_A = [
        [
            sum(di_gamma[t][i][j] for t in range(T - 1)) /
            (sum(gamma[t][i] for t in range(T - 1)) + threshold)
            for j in range(N)
        ]
        for i in range(N)
    ]

    # Re-estimate B
    new_B = [
        [
            sum(gamma[t][j] for t in range(T) if emissions[t] == k) /
            (sum(gamma[t][j] for t in range(T)) + threshold)
            for k in range(M)
        ]
        for j in range(N)
    ]

    return normalize_matrix(new_A), normalize_matrix(new_B), normalize_row(new_pi)


def compute_log_likelihood(scalers):
    return -sum(math.log(s) for s in scalers)

# Baum-Welch algorithm
def baum_welch(A, B, pi, emissions, max_time=5, convergence_threshold=1e-3):
    start_time = time.time()
    iterations = 0
    alpha, scalers = alpha_pass(A, B, pi, emissions)
    beta = beta_pass(A, B, emissions, scalers)
    new_A, new_B, new_pi = re_estimate(A, B, pi, emissions, alpha, beta)
    previous_log_likelihood = compute_log_likelihood(scalers)
    while time.time() - start_time < max_time:
        iterations += 1
        A, B, pi = new_A, new_B, new_pi
        alpha, scalers = alpha_pass(A, B, pi, emissions)
        beta = beta_pass(A, B, emissions, scalers)
        new_A, new_B, new_pi = re_estimate(A, B, pi, emissions, alpha, beta)
        current_log_likelihood = compute_log_likelihood(scalers)
        if abs(current_log_likelihood - previous_log_likelihood) < convergence_threshold:
            break
        previous_log_likelihood = current_log_likelihood
    print(f"Converged in {iterations} iterations")
    return new_A, new_B, new_pi

def randomize_parameters(num_states, num_observations):
    """
    Randomize initial parameters for HMM.
    :param num_states: Number of hidden states
    :param num_observations: Number of observation symbols
    :return: Randomized A, B, and pi
    """
    A = [[random.random() for _ in range(num_states)] for _ in range(num_states)]
    B = [[random.random() for _ in range(num_observations)] for _ in range(num_states)]
    pi = [random.random() for _ in range(num_states)]

    # Normalize rows of A and B, and normalize pi
    A = normalize_matrix(A)
    B = normalize_matrix(B)
    pi = normalize_row(pi)

    return A, B, pi

def matrix_difference(real_matrix, approx_matrix):
    """
    Compute the element-wise difference between the real matrix and the approximated matrix.
    """
    diff_matrix = [[abs(real_matrix[i][j] - approx_matrix[i][j]) for j in range(len(real_matrix[0]))] for i in range(len(real_matrix))]
    return diff_matrix

def print_matrix(matrix, label="Matrix Difference"):
    """
    Print the difference matrix with a label.
    """
    print(f"\n{label}:")
    for row in matrix:
        print(["{:.4f}".format(x) for x in row])

# BIC Score Calculation
def compute_bic_score(A, B, pi, emissions):
    """
    Compute the Bayesian Information Criterion (BIC) score.
    :param A: Transition matrix
    :param B: Emission matrix
    :param pi: Initial probabilities
    :param emissions: Observations
    :return: BIC score
    """
    num_states = len(A)
    num_observations = len(B[0])
    num_parameters = num_states * num_states + num_states * num_observations + num_states - 1
    T = len(emissions)
    log_likelihood = -sum(math.log(x) for x in alpha_pass(A, B, pi, emissions)[1])  # Log-likelihood from alpha_pass
    return -2 * log_likelihood + num_parameters * math.log(T)

# Question 7
def question_7(A, B, pi, emissions_1000, emissions_10000):
    print("\nQuestion 7 Results for T=1000:")
    new_A_1000, new_B_1000, new_pi_1000 = baum_welch(A, B, pi, emissions_1000)
    print(new_A_1000)
    print(new_B_1000)
    print(new_pi_1000)

    print("\nQuestion 7 Results for T=10000:")
    new_A_10000, new_B_10000, new_pi_10000 = baum_welch(A, B, pi, emissions_10000)
    print(new_A_10000)
    print(new_B_10000)
    print(new_pi_10000)

# Question 8
def question_8(A, B, pi, emissions_1000, emissions_10000):
    print("\nQuestion 8 Results with Random Initialization:")

    # Randomize parameters
    random_A, random_B, random_pi = randomize_parameters(num_states=len(A), num_observations=len(B[0]))

    # Train on T=1000
    new_A_1000, new_B_1000, new_pi_1000 = baum_welch(random_A, random_B, random_pi, emissions_1000)

    # Train on T=10000
    new_A_10000, new_B_10000, new_pi_10000 = baum_welch(random_A, random_B, random_pi, emissions_10000)

    # Compare with real matrices
    diff_A_1000 = matrix_difference(A, new_A_1000)
    diff_B_1000 = matrix_difference(B, new_B_1000)
    diff_A_10000 = matrix_difference(A, new_A_10000)
    diff_B_10000 = matrix_difference(B, new_B_10000)
    print(diff_A_1000)
    print_matrix_difference(diff_A_1000, label="A Difference (T=1000)")
    print_matrix_difference(diff_B_1000, label="B Difference (T=1000)")
    print_matrix_difference(diff_A_10000, label="A Difference (T=10000)")
    print_matrix_difference(diff_B_10000, label="B Difference (T=10000)")


# Question 9
def question_9(emissions_1000, emissions_10000, max_hidden_states=6):
    """
    Explore different numbers of hidden states and compare results.
    :param emissions_1000: Observations for T=1000
    :param emissions_10000: Observations for T=10000
    :param max_hidden_states: Maximum number of hidden states to test
    """
    print("\nQuestion 9 Results:")
    num_observations = 4  # Fixed number of observation symbols

    for num_states in range(1, max_hidden_states + 1):
        print(f"\nNumber of hidden states: {num_states}")

        # Randomize initial parameters
        random_A, random_B, random_pi = randomize_parameters(num_states, num_observations)

        # Train on T=1000
        new_A_1000, new_B_1000, new_pi_1000 = baum_welch(random_A, random_B, random_pi, emissions_1000)

        # Train on T=10000
        new_A_10000, new_B_10000, new_pi_10000 = baum_welch(random_A, random_B, random_pi, emissions_10000)

        # Compute BIC scores
        bic_1000 = compute_bic_score(new_A_1000, new_B_1000, new_pi_1000, emissions_1000)
        bic_10000 = compute_bic_score(new_A_10000, new_B_10000, new_pi_10000, emissions_10000)

        print(f"BIC Score for T=1000: {bic_1000:.4f}")
        print(f"BIC Score for T=10000: {bic_10000:.4f}")

def question_10(emissions, real_A, real_B, real_pi):
    """
    Test the Baum-Welch algorithm with three different initialization strategies:
    1. Uniform distribution
    2. Diagonal A matrix and pi = [0, 0, 1]
    3. Matrices close to the solution
    """

    print("\n### Question 10: Testing Different Initializations ###")

    # Uniform Distribution Initialization
    print("\n--- Uniform Distribution Initialization ---")
    num_states = len(real_A)
    num_observations = len(real_B[0])
    uniform_A = [[1 / num_states for _ in range(num_states)] for _ in range(num_states)]
    uniform_B = [[1 / num_observations for _ in range(num_observations)] for _ in range(num_states)]
    uniform_pi = [1 / num_states for _ in range(num_states)]

    new_A_uniform, new_B_uniform, new_pi_uniform = baum_welch(uniform_A, uniform_B, uniform_pi, emissions)
    print_matrix(new_A_uniform, label="Learned A Matrix (Uniform)")
    print_matrix(new_B_uniform, label="Learned B Matrix (Uniform)")
    print("Learned Pi (Uniform):", new_pi_uniform)

    # Diagonal A Matrix and pi = [0, 0, 1]
    print("\n--- Diagonal A Matrix and Pi = [0, 0, 1] ---")
    diagonal_A = [[1 if i == j else 0 for j in range(num_states)] for i in range(num_states)]
    diagonal_pi = [0] * (num_states - 1) + [1]

    new_A_diag, new_B_diag, new_pi_diag = baum_welch(diagonal_A, real_B, diagonal_pi, emissions)
    print_matrix(new_A_diag, label="Learned A Matrix (Diagonal)")
    print_matrix(new_B_diag, label="Learned B Matrix (Diagonal)")
    print("Learned Pi (Diagonal):", new_pi_diag)

    # Initialization Close to the Solution
    print("\n--- Initialization Close to the Solution ---")
    close_A = [[real_A[i][j] + random.uniform(-0.1, 0.1) for j in range(num_states)] for i in range(num_states)]
    close_B = [[real_B[i][j] + random.uniform(-0.1, 0.1) for j in range(num_observations)] for i in range(num_states)]
    close_pi = [real_pi[i] + random.uniform(-0.1, 0.1) for i in range(num_states)]

    # Normalize the randomized parameters
    close_A = normalize_matrix(close_A)
    close_B = normalize_matrix(close_B)
    close_pi = normalize_row(close_pi)

    new_A_close, new_B_close, new_pi_close = baum_welch(close_A, close_B, close_pi, emissions)
    print_matrix(new_A_close, label="Learned A Matrix (Close to Solution)")
    print_matrix(new_B_close, label="Learned B Matrix (Close to Solution)")
    print("Learned Pi (Close to Solution):", new_pi_close)

# Main function
def main():
    A = [[0.54, 0.26, 0.20], [0.19, 0.53, 0.28], [0.22, 0.18, 0.6]]
    B = [[0.5, 0.2, 0.11, 0.19], [0.22, 0.28, 0.23, 0.27], [0.19, 0.21, 0.15, 0.45]]
    pi = [0.3, 0.2, 0.5]

    with open("hmm_c_N1000.in") as f:
        emissions_1000 = list(map(int, f.read().split()))
    with open("hmm_c_N10000.in") as f:
        emissions_10000 = list(map(int, f.read().split()))

    valid_emissions = {0, 1, 2, 3}
    emissions_1000 = [e for e in emissions_1000 if e in valid_emissions]
    emissions_10000 = [e for e in emissions_10000 if e in valid_emissions]

    #question_7(A, B, pi, emissions_1000, emissions_10000)
    question_8(A, B, pi, emissions_1000, emissions_10000)
    #question_9(emissions_1000, emissions_10000)
    #question_10(emissions_1000, A, B, pi)


if __name__ == "__main__":
    main()

