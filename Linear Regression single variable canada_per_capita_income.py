#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[52]:


df = pd.read_csv(r'C:\DS practice\canada_per_capita_income.csv')


# In[86]:


df.head()


# In[106]:


plt.xlabel('year')
plt.ylabel('income')
plt.scatter(df.year,df.income)


# In[131]:


reg = linear_model.LinearRegression()


# In[135]:


X = df[['year']].values
y = df.income


# In[136]:


reg.fit(X,y)


# In[138]:


reg.predict([[2020]])  #2D array error


# In[112]:


plt.xlabel('year')
plt.ylabel('income')
plt.scatter(df.year,df.income)
plt.plot(df.year,reg.predict(df[['year']]) )


# In[114]:


reg.coef_


# In[120]:


reg.intercept_


# In[121]:


828.46507522*2020+(-1632210.7578554575)


# In[123]:


import joblib


# In[126]:


joblib.dump(reg,'C:\DS practice\modelLinearRegressionSingleVariable_joblib')


# In[128]:


mj = joblib.load('C:\DS practice\modelLinearRegressionSingleVariable_joblib')


# In[139]:


mj.predict([[2020]])


# In[ ]:




