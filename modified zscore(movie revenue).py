#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv(r'C:\DS practice\movie_revenues.csv')


# In[6]:


df.head()


# In[9]:


df.revenue.describe()


# In[14]:


df['revenue_mil'] = df.revenue.apply(lambda x: x/1000000)


# In[15]:


df.revenue_mil.describe()


# In[16]:


_, mean, std, *_ = df.revenue_mil.describe()


# In[17]:


mean,std


# In[18]:


def get_z_score(value,mean,std):
    return (value-mean)/std


# In[21]:


df['zscore'] = df.revenue_mil.apply(lambda x: get_z_score(x,mean,std))


# In[28]:


df[df['zscore'] > 4]


# In[36]:


def get_mad(s):
    median = np.median(s)
    diff = abs(s-median)
    MAD = np.median(diff)
    return MAD


# In[37]:


MAD = get_mad(df.revenue_mil)
median = np.median(df.revenue_mil)
MAD,median


# In[42]:


def get_modified_zscore(x,median,MAD):
    return 0.6745*(x-median)/MAD


# In[43]:


get_modified_zscore(2787,median,MAD)


# In[45]:


df['mod_zscore'] = df.revenue_mil.apply(lambda x: get_modified_zscore(x,median,MAD))


# In[49]:


df[df.mod_zscore > 3.5]


# In[ ]:




