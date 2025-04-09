def read_matrix():
    line = input().strip().split()
    rows, cols = int(line[0]), int(line[1])
    elements = list(map(float, line[2:]))
    matrix = [elements[i * cols:(i + 1) * cols] for i in range(rows)]
    return matrix

def matrix_multiply(vector, matrix):
    """
    Multiplies a row vector with a matrix.
    """
    result = []
    for col in range(len(matrix[0])): 
        sum_value = 0
        for row in range(len(matrix)):
            sum_value += vector[row] * matrix[row][col]
        result.append(sum_value)
    return result

def format_output(probabilities):
    """
    Formats the output as required by the problem:
    - Indices followed by the probabilities as space-separated values.
    """
    num_rows = 1
    num_cols = len(probabilities)
    result = f"{num_rows} {num_cols} " + " ".join(map(str, probabilities))
    return result

if __name__ == "__main__":
    A = read_matrix()  
    B = read_matrix() 
    current_state = read_matrix()[0]  
    next_state = matrix_multiply(current_state, A)
    emissions = matrix_multiply(next_state, B)
    output = format_output(emissions)
    print(output)