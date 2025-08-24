import pandas as pd

df = pd.read_csv('merged_ctu13.csv', low_memory=False)
print(df.columns.tolist())  # 查看真实列名

