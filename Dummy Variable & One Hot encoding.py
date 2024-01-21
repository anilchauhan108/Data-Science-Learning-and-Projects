import pandas as pd
import numpy as np

df = pd.read_csv('C:\DS practice\carprices.csv')
df

dummy = pd.get_dummies(df['Car Model'])
dummy

merged = pd.concat([df, dummy], axis = 'columns')
merged

final = merged.drop(['Car Model', 'Mercedez Benz C class'], axis = 'columns')
final

from sklearn.linear_model import LinearRegression
model = LinearRegression()

X = final.drop('Sell Price($)', axis = 'columns')
y = final['Sell Price($)']


model.fit(X,y)


model.predict([[45000,4,0,0]])


model.predict([[86000,4,0,1]])

model.score(X,y)

from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()

dfle = df
dfle['Car Model'] = le.fit_transform(dfle['Car Model'])
dfle



X = dfle[['Car Model', 'Mileage', 'Age(yrs)']].values
X


y = dfle['Sell Price($)'].values


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


import scipy.sparse

def is_dense(X):
    return isinstance(X, np.ndarray)

ct = ColumnTransformer([("Car Model", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
X                  


X = X[:,1:]
X


model.fit(X,y)


model.predict([[0,1,45000,4]])


model.predict([[0,1,86000,7]])


model.score(X,y)







