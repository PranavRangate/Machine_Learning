import pandas as pd
import numpy as np
data=pd.read_csv('d.csv')
x = data.iloc[:,:-1]

y = data.iloc[:,-1]

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
imputer.fit(x[1:])
x[1:]=imputer.transform(x[1:])
x
