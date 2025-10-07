def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    # Return a list where each element is the dot product of a row of 'a' with 'b'.
    # If the number of columns in 'a' does not match the length of 'b', return -1.
    num_cols = len(a[0]) if a else 0

    if num_cols != len(b):
        return -1

    result_vector = []

    for row in a:
        dot_product_sum = sum(num_val * vec_val for num_val, vec_val in zip(row, b))
        result_vector.append(dot_product_sum)

    return result_vector

if __name__ == "__main__":
    # Example usage:
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    vector = [1, 0, -1]
    result = matrix_dot_vector(matrix, vector)
    print("Result of matrix-vector dot product:", result)