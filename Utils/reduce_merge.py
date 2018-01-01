from functools import reduce
import pandas as pd
import numpy as np

# Merge multiple dataframes
df1 = pd.DataFrame(np.array([
    ['a', 5, 9],
    ['b', 4, 61],
    ['c', 24, 9]]),
    columns=['name', 'attr11', 'attr12'])
df2 = pd.DataFrame(np.array([
    ['a', 5, 19],
    ['b', 14, 16],
    ['c', 4, 9]]),
    columns=['name', 'attr21', 'attr22'])
df3 = pd.DataFrame(np.array([
    ['a', 15, 49],
    ['b', 4, 36],
    ['c', 14, 9]]),
    columns=['name', 'attr31', 'attr32'])

##df = pd.merge(pd.merge(df1,df2,on='name'),df3,on='name', how='outer')


df_lst = [df1, df2, df3]
df = reduce(lambda left, right: pd.merge(left, right, on='name', how='outer'),
            df_lst)

print(df1)
print(df2)
print(df3)
print(df)
