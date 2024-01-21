#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib
import seaborn as sn
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


df = pd.read_csv(r'C:\DS practice\bhp.csv')


# In[20]:


df.head()


# In[40]:


df['price_per_sqft'].describe()


# In[33]:


lower_limit, upper_limit = df['price_per_sqft'].quantile([0.001,0.999])
lower_limit, upper_limit


# In[42]:


sn.histplot(df.price_per_sqft,bins = 20, kde = True)


# In[80]:


Outliers = df[(df.price_per_sqft < lower_limit) | (df.price_per_sqft > upper_limit)]
Outliers.sample(10)


# In[45]:


df2 = df[(df.price_per_sqft > lower_limit) & (df.price_per_sqft < upper_limit)]


# In[46]:


sn.histplot(df2.price_per_sqft,bins = 20, kde = True)


# In[47]:


df.shape


# In[48]:


df2.shape


# In[50]:


df.shape[0] - df2.shape[0]


# In[64]:


max_limit = df2.price_per_sqft.mean() + 4*df2.price_per_sqft.std()
min_limit = df2.price_per_sqft.mean() - 4*df2.price_per_sqft.std()


# In[72]:


Outliers2 = df2[(df2.price_per_sqft < min_limit) | (df2.price_per_sqft > max_limit)]


# In[74]:


Outliers2


# In[70]:


df3 = df2[(df2.price_per_sqft > min_limit) & (df2.price_per_sqft < max_limit)]


# In[71]:


df2.shape[0] - df3.shape[0]


# In[81]:


df3.shape


# In[84]:


sn.histplot(df3.price_per_sqft, kde = True, bins = 20)


# In[92]:


df2['zscore'] = (df2.price_per_sqft - df2.price_per_sqft.mean())/df2.price_per_sqft.std()
df2.sample(10)


# In[103]:


Outlier3 = df2[(df2.zscore < -4) | (df2.zscore > 4)]


# In[104]:


Outlier3


# In[105]:


df4 = df2[(df2.zscore > -4) & (df2.zscore < 4)]


# In[106]:


sn.histplot(df4.price_per_sqft, kde = True, bins = 20)


# In[ ]:




