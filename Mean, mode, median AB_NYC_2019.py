#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[9]:


df = pd.read_csv(r'C:\DS practice\AB_NYC_2019.csv')


# In[16]:


df['price'].describe()


# In[87]:


df2 = df[df.price < df.price.quantile(0.99)]


# In[89]:


df2.price.describe()


# In[93]:


df.drop('price', axis = 'columns')


# In[94]:


df['price'] = df2.price


# In[96]:


df.price.describe()


# In[98]:


df.head()


# In[99]:


df['price'][2] = np.NaN


# In[100]:


df.head()


# In[102]:


df.fillna(df.price.median())


# In[106]:


df.price.median()


# In[ ]:




