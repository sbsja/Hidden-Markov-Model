def read_matrix():
    line = input().strip().split()
    rows, cols = int(line[0]), int(line[1])
    elements = list(map(float, line[2:]))
    return [elements[i * cols:(i + 1) * cols] for i in range(rows)]

def viterbi_algorithm(A, B, pi, emissions):
    """
    Implements the Viterbi Algorithm to find the most likely sequence of states.
    :param A: Transition matrix
    :param B: Emission matrix
    :param pi: Initial state probability distribution
    :param emissions: List of observed emissions
    :return: List of the most probable sequence of states
    """

    T = len(emissions)  # Length of the observed emission sequence
    N = len(pi)  # Number of states in the HMM

    # Initialize delta table to store the highest probability of any path to each state at each time step
    delta = [[0] * N for i in range(T)]
    
    # Initialize backpointer table to store the most probable previous state for each state at each time step
    backpointer = [[0] * N for i in range(T)]

    # Initialization step (time step t = 0)
    for i in range(N):
        delta[0][i] = pi[i] * B[i][emissions[0]]  # Compute initial probabilities for each state
        backpointer[0][i] = 0  # No predecessor for the first state

    # (time steps t = 1 to T-1)
    for t in range(1, T):  # Iterate over each time step
        for current_state in range(N):  # Iterate over each possible current state
            max_probability = 0  # Initialize the maximum probability for the current state
            best_previous_state = 0  # Initialize the best previous state for the current state

            for previous_state in range(N):  # Iterate over each possible previous state
                # Compute the probability of transitioning from the previous state to the current state
                transition_probability = delta[t-1][previous_state] * A[previous_state][current_state]

                # Update max_probability and best_previous_state if this path is more likely
                if transition_probability > max_probability:
                    max_probability = transition_probability
                    best_previous_state = previous_state

            # Update delta for the current state with the maximum probability path
            delta[t][current_state] = max_probability * B[current_state][emissions[t]]
            # Record the best previous state in the backpointer table
            backpointer[t][current_state] = best_previous_state

    # Termination step: Find the state with the highest probability at the final time step
    max_final_prob = 0  # Initialize the maximum probability of the final state
    final_state = 0  # Initialize the final state index
    for state in range(N):  # Iterate over all states
        if delta[T-1][state] > max_final_prob:  # Check if this state has the highest probability
            max_final_prob = delta[T-1][state]
            final_state = state

    # Backtracking step: Reconstruct the most likely sequence of states
    most_likely_sequence = [0] * T  # Initialize the sequence with placeholders
    most_likely_sequence[-1] = final_state  # Start with the final state
    for t in range(T-2, -1, -1):  # Iterate backwards through time steps
        # Use the backpointer to find the previous state in the most likely path
        most_likely_sequence[t] = backpointer[t+1][most_likely_sequence[t+1]]

    print(backpointer)

    return most_likely_sequence  # Return the most likely sequence of states


if __name__ == "__main__":
    A = read_matrix() 
    B = read_matrix()  
    pi = read_matrix()[0]
    emissions_input = input().strip().split()
    emissions = list(map(int, emissions_input[1:])) 
    most_likely_sequence = viterbi_algorithm(A, B, pi, emissions)
    print(" ".join(map(str, most_likely_sequence)))
