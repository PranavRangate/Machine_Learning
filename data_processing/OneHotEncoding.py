import pandas as pd 
import numpy as np

data = pd.read_csv('Data1.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')

x = np.array(ct.fit_transform(x))

print(x)
