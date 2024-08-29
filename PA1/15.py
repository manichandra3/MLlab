import numpy as np
from collections import Counter

np.random.seed(42)
matrix = np.random.randint(1, 6, size=(10, 5))
print("Random 10x5 Matrix:")
print(matrix)
for i in range(5):
    feature_counts = Counter(matrix[:, i])
    print(f"\nFeature {i+1} counts:")
    for value, count in feature_counts.items():
        print(f"Value {value}: {count} patterns")
        