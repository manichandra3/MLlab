import numpy as np

np.random.seed(42)
matrix = np.random.rand(10, 5) * 100
matrix = np.round(matrix, 2)
max_values = np.max(matrix, axis=0)
min_values = np.min(matrix, axis=0)

print("Random 10x5 Matrix:")
print(matrix)
print("\nMaximum values for each feature:")
print(max_values)
print("\nMinimum values for each feature:")
print(min_values)
