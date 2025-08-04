"""
A simple file that I used to manually check data. Inefficient, but it works for me.
"""

import pandas as pd
df = pd.read_parquet("graphrag_workspace/output/entities.parquet")

print(df.iloc[1])



