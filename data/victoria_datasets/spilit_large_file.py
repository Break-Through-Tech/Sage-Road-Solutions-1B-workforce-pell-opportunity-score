import pandas as pd

df = pd.read_csv("data/victoria_datasets/Most-Recent-Cohorts-Field-of-Study.csv", low_memory=False)

n_chunks = 3  # 146 MB / 3 ≈ 49 MB per file, safely under 100 MB
chunk_size = len(df) // n_chunks + 1

for i in range(n_chunks):
    start = i * chunk_size
    end = start + chunk_size
    chunk = df.iloc[start:end]
    chunk.to_csv(f"data/victoria_datasets/Field-of-Study_part{i+1}.csv", index=False)
    print(f"part{i+1}: {len(chunk)} rows")