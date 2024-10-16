import pandas as pd

df=pd.read_csv('Churn.csv')

print(df.shape)

df.drop('Churn.csv',axis='column',inplace=True)
print(df.size)