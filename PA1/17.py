import pandas as pd
import sys

df = pd.read_csv('output_matrix.csv', header=0)
print(df.shape)
df_trimmed = df.iloc[:-1, :-1]
print(df_trimmed.shape)
