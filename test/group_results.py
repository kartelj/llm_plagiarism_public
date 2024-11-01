import pandas as pd

df = pd.read_csv("results.csv")
grouped_df = df.groupby(['language', 'features', 'domain', 'same_length', 'source', 'temperature'])['accuracy'].mean().reset_index()
grouped_df.to_csv('grouped_except_algorithm.csv', index=False)

grouped_df = df.groupby(['language', 'features', 'same_length', 'temperature'])['accuracy'].mean().reset_index()
grouped_df.to_csv('grouped_length_temperature.csv', index=False)

grouped_df = df.groupby(['language', 'features', 'domain', 'source'])['accuracy'].mean().reset_index()
grouped_df.to_csv('grouped_domain_source.csv', index=False)