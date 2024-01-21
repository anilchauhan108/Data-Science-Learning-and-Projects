#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib 
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv(r'C:\DS practice\revenue.csv')


# In[10]:


df.plot(x = 'company', y = 'revenue', kind = 'bar')


# In[12]:


df.plot(x = 'company', y = 'revenue', kind = 'bar', logy = True) #similarly logx can be used to normalize people
#earning(jeff bezos right skewed data example)


# In[13]:


df2 = pd.read_csv(r'C:\DS practice\income.csv')


# In[14]:


df2


# In[ ]:





# In[ ]:




