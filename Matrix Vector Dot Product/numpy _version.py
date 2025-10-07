import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float] | int:
  """
  Computes the dot product of a matrix 'a' and a vector 'b' using NumPy.
  
  Returns a list where each element is the dot product of a row of 'a' with 'b'.
  If the number of columns in 'a' does not match the length of 'b', it returns -1.

  Args:
    a: A matrix represented as a list of lists of numbers (int or float).
    b: A vector represented as a list of numbers (int or float).

  Returns:
    A list representing the resulting vector if the dimensions are compatible.
    Returns -1 if the dimensions are incompatible for a dot product.
  """
  # Determine the number of columns from the first row if the matrix is not empty.
  # If the matrix is empty (0 rows), the number of columns is considered 0.
  num_cols = len(a[0]) if a else 0

  # If the matrix is empty (a has 0 rows), the dot product is a valid
  # operation only if vector 'b' is also empty (has length 0).
  # The result of a 0xN * N-vector is a 0-vector (empty list).
  if not a:
    return [] if not b else -1

  # The core requirement for matrix-vector multiplication:
  # The number of columns in the matrix must equal the number of elements (length) in the vector.
  if num_cols != len(b):
    return -1
  
  # If the dimensions are compatible, convert to NumPy arrays and compute the dot product.
  # NumPy handles these calculations very efficiently.
  matrix_np = np.array(a)
  vector_np = np.array(b)
  
  result_vector = np.dot(matrix_np, vector_np)
  
  # Convert the resulting NumPy array back to a standard Python list before returning.
  return result_vector.tolist()

# Example 1: Valid multiplication (from the problem description)
matrix_a = [[1, 2], [2, 4]]
vector_b = [1, 2]
print(f"Input Matrix: {matrix_a}")
print(f"Input Vector: {vector_b}")
print(f"Dot Product: {matrix_dot_vector(matrix_a, vector_b)}")
# Expected output: [5, 10]

print("-" * 20)

# Example 2: Incompatible dimensions
matrix_c = [[1, 2, 3], [4, 5, 6]] # A 2x3 matrix
vector_d = [1, 2]                 # A vector of length 2
print(f"Input Matrix: {matrix_c}")
print(f"Input Vector: {vector_d}")
print(f"Dot Product: {matrix_dot_vector(matrix_c, vector_d)}")
# Expected output: -1

print("-" * 20)

# Example 3: Another valid multiplication
matrix_e = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # 3x3 Identity matrix
vector_f = [5, 10, 15]                        # A vector of length 3
print(f"Input Matrix: {matrix_e}")
print(f"Input Vector: {vector_f}")
print(f"Dot Product: {matrix_dot_vector(matrix_e, vector_f)}")
# Expected output: [5, 10, 15]
