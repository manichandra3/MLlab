import numpy as np

# I
# col1: no.of hours of sun shine.
# col2: no. of ice creams sold.

data_set = np.array([[4, 2], [5, 3], [7, 5], [10, 7], [15, 9], [12, 8], [18, 11], [20, 14], [22, 16], [25, 19]])

print("First five rows:")
print(data_set[:5])
print("No. of rows: " + str(data_set.shape[0]))
print("No. of cols: " + str(data_set.shape[1]))
print("Range of values for feature 1 (hours of sunshine): ")
print(data_set[:, 0].min(), data_set[:, 0].max())
print("Range of values for feature 2 (ice creams sold): ")
print(data_set[:, 1].min(), data_set[:, 1].max())


# II
# Randomly split the dataset
def random_split(data, ratio_x, ratio_y):
    size = data.shape[0]
    size_x = int(size * ratio_x)
    size_y = int(size * ratio_y)
    all_indices = np.arange(size)
    np.random.shuffle(all_indices)
    indices_y = all_indices[:size_y]
    indices_x = all_indices[size_y:size_y + size_x]
    data_x = data[indices_x]
    data_y = data[indices_y]
    return data_x, data_y


data_x_1, data_y_1 = random_split(data_set, 0.7, 0.3)
data_x_2, data_y_2 = random_split(data_set, 0.8, 0.2)
data_x_3, data_y_3 = random_split(data_set, 0.9, 0.1)

print("Split 1:")
print("data_x_1 shape:", data_x_1.shape)
print("data_y_1 shape:", data_y_1.shape)

print("Split 2:")
print("data_x_2 shape:", data_x_2.shape)
print("data_y_2 shape:", data_y_2.shape)

print("Split 3:")
print("data_x_3 shape:", data_x_3.shape)
print("data_y_3 shape:", data_y_3.shape)
