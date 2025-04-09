import math
import time

def normalize_row(row, threshold=1e-8):
    """
    Normalize a single row and set values below the threshold to zero.
    """
    row = [x if x > threshold else 0 for x in row]
    row_sum = sum(row)
    return [x / row_sum for x in row] if row_sum > 0 else row

def normalize_matrix(matrix, threshold=1e-8):
    """
    Normalize an entire matrix row-wise and set values below the threshold to zero.
    """
    return [normalize_row(row, threshold) for row in matrix]

def alpha_pass(A, B, pi, emissions):
    """
    Computes the probability of the given emission sequence using the forward algorithm.
    :param A: Transition matrix
    :param B: Emission matrix
    :param pi: Initial state distribution
    :param emissions: List of observed emissions
    :return: Total probability of the emission sequence
    """
    alpha = []
    scalers = []  # Introduced to avoid underflow

    # t=0
    alpha.append([pi[0][i] * B[i][emissions[0]] for i in range(len(pi[0]))])
    scalers.append(1/sum(alpha[0]))
    alpha[0] = [alpha_0_i * scalers[0] for alpha_0_i in alpha[0]]

    # t>0
    for t in range(1, len(emissions)):
        alpha.append([sum(alpha[t-1][j] * A[j][i] for j in range(len(A))) * B[i][emissions[t]] for i in range(len(A))])
        scalers.append(1/sum(alpha[t]))
        alpha[t] = [alpha_t_i * scalers[t] for alpha_t_i in alpha[t]]

    return alpha, scalers

def beta_pass(A, B, emissions, scalers):
    """
    Compute the backward probabilities using the Backward Algorithm.
    """
    beta = []
    beta.append([scalers[-1] for _ in range(len(A))])
    for t in range(len(emissions)-2, -1, -1):
        beta.insert(0, [sum(A[i][j] * B[j][emissions[t+1]] * beta[0][j] for j in range(len(A))) for i in range(len(A))])
        beta[0] = [beta_0_i * scalers[t] for beta_0_i in beta[0]]
    return beta

def get_gammas(A, B, alpha, beta, emissions):
    """
    Compute the gamma and di-gamma values for the Baum-Welch algorithm.
    """
    gamma = []
    di_gamma = []

    for t in range(len(emissions)-1):
        di_gamma.append([[alpha[t][i] * A[i][j] * B[j][emissions[t+1]] * beta[t+1][j] for j in range(len(A))] for i in range(len(A))])
        gamma.append([sum(di_gamma[t][i]) for i in range(len(A))])
    gamma.append(alpha[-1])
    return gamma, di_gamma

def re_estimate(A, B, pi, emissions, alpha, beta):
    """
    Re-estimate the parameters A, B, and pi based on the computed gamma values.
    """

    N = len(A)
    M = len(B[0])
    T = len(emissions)

    gamma, di_gamma = get_gammas(A, B, alpha, beta, emissions)

    # Re-estimate pi
    new_pi = [[gamma[0][i] for i in range(N)]]

    # Re-estimate A
    new_A = [[sum(di_gamma[t][i][j] for t in range(T-1)) / sum(gamma[t][i] for t in range(T-1)) for j in range(N)] for i in range(N)]

    # Re-estimate B
    new_B = [[sum(gamma[t][j] for t in range(T) if emissions[t] == k) / sum(gamma[t][j] for t in range(T)) for k in range(M)] for j in range(N)]

    # Normalize with threshold
    new_A = normalize_matrix(new_A)
    new_B = normalize_matrix(new_B)
    
    return new_A, new_B, new_pi

def compute_log_likelihood(scalers):
    """
    Compute the log likelihood of the emission sequence using the scaling factors.
    """
    return -sum(math.log(s) for s in scalers)

def baum_welch(A, B, pi, emissions, max_time=0.8):
    """
    Perform the Baum-Welch algorithm for training a Hidden Markov Model (HMM).
    :param A: Transition probability matrix
    :param B: Emission probability matrix
    :param pi: Initial state probability distribution
    :param emissions: List of observed emissions
    :param max_time: Maximum allowed execution time in seconds
    :return: Updated transition matrix (A), emission matrix (B), and initial state probabilities (pi)
    """

    # Compute forward probabilities (alpha) and scaling factors to prevent underflow
    alpha, scalers = alpha_pass(A, B, pi, emissions)
    
    # Compute backward probabilities (beta)
    beta = beta_pass(A, B, emissions, scalers)
    
    # Estimate new model parameters (A, B, pi) using alpha and beta
    new_A, new_B, new_pi = re_estimate(A, B, pi, emissions, alpha, beta)
    
    # Initialize log-likelihood tracking with a very low value
    previous_log_likelihood = float("-inf")

    # Record start time for execution timing
    start_time = time.time()
    
    # Iterate until convergence or time limit is reached
    while True:
        # Update model parameters
        A, B, pi = new_A, new_B, new_pi
        
        # Compute the log-likelihood of the current model
        current_log_likelihood = compute_log_likelihood(scalers)

        # Stop if log-likelihood decreases (convergence) or time exceeds the limit
        if previous_log_likelihood > current_log_likelihood or time.time() - start_time > max_time:
            break

        # Update the previous log-likelihood
        previous_log_likelihood = current_log_likelihood
        
        # Recompute forward probabilities with updated parameters
        alpha, scalers = alpha_pass(A, B, pi, emissions)
        
        # Recompute backward probabilities with updated parameters
        beta = beta_pass(A, B, emissions, scalers)
        
        # Re-estimate model parameters again
        new_A, new_B, new_pi = re_estimate(A, B, pi, emissions, alpha, beta)

    # Return the final estimated parameters
    return new_A, new_B, new_pi

def get_output(A):
    """
    Convert a matrix into a formatted string representation.
    """
    out = ""
    out += str(len(A)) + " " + str(len(A[0])) + " "
    for row in A:
        out += " ".join(map(str, row)) + " "
    return out

def read_line(line_char):
    line = line_char.split()
    n = int(line.pop(0))
    k = int(line.pop(0))
    return [[float(line.pop(0)) for _ in range(k)] for _ in range(n)]

def read_obs(line_char):
    line = line_char.split()
    return [int(i) for i in line[1:]]

def read_input():
    lines = []
    while True:
        try:
            lines.append(input())
        except EOFError:
            break
    return [read_line(line) for line in lines[:3]] + [read_obs(lines[3])]

def main():
    A, B, pi, emissions = read_input()

    new_A, new_B, new_pi = baum_welch(A, B, pi, emissions)

    print(get_output(new_A))
    print(get_output(new_B))

if __name__ == "__main__":
    main()
