import pandas as pd 
import numpy as np

data = pd.read_csv('Data1.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder

print(y)

le = LabelEncoder()

y = le.fit_transform(y)
print('After label encoding on y section ')
print(y)