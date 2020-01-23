import pandas as pd
import numpy as np

df = pd.read_csv('../creditcard.csv')
print(df.info())
print(df.Class.value_counts())
