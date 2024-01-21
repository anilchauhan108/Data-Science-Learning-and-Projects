#!/usr/bin/env python
# coding: utf-8

# In[220]:


import pandas as pd
import numpy as np


# In[221]:


df = pd.read_csv('C:\DS practice\carprices.csv')


# In[222]:


df


# In[223]:


dummy = pd.get_dummies(df['Car Model'])
dummy


# In[224]:


merged = pd.concat([df, dummy], axis = 'columns')
merged


# In[225]:


final = merged.drop(['Car Model', 'Mercedez Benz C class'], axis = 'columns')
final


# In[226]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[227]:


X = final.drop('Sell Price($)', axis = 'columns')
y = final['Sell Price($)']


# In[228]:


model.fit(X,y)


# In[229]:


model.predict([[45000,4,0,0]])


# In[230]:


model.predict([[86000,4,0,1]])


# In[231]:


model.score(X,y)


# In[232]:


from sklearn.preprocessing import LabelEncoder


# In[233]:


le = LabelEncoder()


# In[234]:


dfle = df
dfle['Car Model'] = le.fit_transform(dfle['Car Model'])
dfle


# In[235]:


X = dfle[['Car Model', 'Mileage', 'Age(yrs)']].values
X


# In[236]:


y = dfle['Sell Price($)'].values


# In[237]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[238]:


import scipy.sparse

def is_dense(X):
    return isinstance(X, np.ndarray)


# In[239]:


ct = ColumnTransformer([("Car Model", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
X                  


# In[240]:


X = X[:,1:]
X


# In[241]:


model.fit(X,y)


# In[242]:


model.predict([[0,1,45000,4]])


# In[245]:


model.predict([[0,1,86000,7]])


# In[244]:


model.score(X,y)


# In[ ]:




