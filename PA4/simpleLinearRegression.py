import numpy as np
import pandas as pd

def random_split(data, ratio_x, ratio_y):
    size = data.shape[0]
    size_x = int(size * ratio_x)
    size_y = int(size * ratio_y)
    all_indices = np.arange(size)
    np.random.shuffle(all_indices)
    indices_y = all_indices[:size_y]
    indices_x = all_indices[size_y:size_y + size_x]
    data_x = data.iloc[indices_x]
    data_y = data.iloc[indices_y]
    return data_x, data_y

data = pd.read_csv('data.csv')
data_x, data_y = random_split(data, 0.7, 0.3)
print(f"data_x:\n{data_x}\n\ndata_y:\n{data_y}")
