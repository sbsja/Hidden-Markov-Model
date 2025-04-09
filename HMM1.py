def read_matrix():
    line = input().strip().split()
    rows, cols = int(line[0]), int(line[1])
    elements = list(map(float, line[2:]))
    matrix = [elements[i * cols:(i + 1) * cols] for i in range(rows)]
    return matrix


def forward_algorithm(A, B, pi, emissions):
    """
    Computes the probability of the given emission sequence using the forward algorithm.
    :param A: Transition matrix
    :param B: Emission matrix
    :param pi: Initial state distribution
    :param emissions: List of observed emissions
    :return: Total probability of the emission sequence
    """
    N = len(pi)  # Number of states in the HMM
    T = len(emissions)  # Length of the observed emission sequence

    # Step 1: Initialization
    # Compute the initial forward probabilities for each state (time step t = 0)
    alpha = [pi[i] * B[i][emissions[0]] for i in range(N)]

    # Iterate over each subsequent time step (t = 1 to T-1)
    for t in range(1, T):
        new_alpha = [0] * N  # Create a new alpha array for time step t
        for i in range(N):
            # Compute the forward probability for state i at time t
            # Sum over all previous states (j) at time t-1
            new_alpha[i] = sum(alpha[j] * A[j][i] for j in range(N)) * B[i][emissions[t]]
        alpha = new_alpha  # Update alpha for the next iteration
        print(alpha)

    # Step 3: Termination
    # The total probability of the emission sequence is the sum of the final alpha values
    return sum(alpha)

if __name__ == "__main__":
    A = read_matrix()  
    B = read_matrix()  
    pi = read_matrix()[0]
    emissions_input = input().strip().split()
    T = int(emissions_input[0])
    emissions = list(map(int, emissions_input[1:]))
    result = forward_algorithm(A, B, pi, emissions)
    print(result)
