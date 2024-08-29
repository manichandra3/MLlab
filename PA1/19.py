import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('salary_data.csv')

output_range = (df.iloc[:, -1].min(), df.iloc[:, -1].max())
print("\nnumber of features:")
print(df.shape[1])
print("\nnumber of patterns:")
print(df.shape[0])
print("\nrange:", output_range)


# train_test split?
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


for train_size in np.arange(0.1, 1.0, 0.1):
    print(f"\nTrain:Test Ratio = {train_size * 100:.0f}:{(1 - train_size) * 100:.0f}")
    split_dataset(df, train_size)
