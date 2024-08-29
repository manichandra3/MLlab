import pandas as pd

df = pd.read_csv('salary_data.csv')
print(df)

print(df.mean(axis=0))
print(df.std(axis=0))
print(df.median(axis=0))
print(df.mean(axis=1))
print(df.std(axis=1))
print(df.median(axis=1))
print(df.mode(axis=0))
