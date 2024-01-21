#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[3]:


df = pd.read_csv('C:\DS practice\hiring.csv')


# In[5]:


df


# In[6]:


pip install word2number


# In[7]:


from word2number import w2n


# In[13]:


df.experience = df.experience.fillna('zero') #fills NaN value in "experience" column to zero


# In[19]:


df.experience = df.experience.apply(w2n.word_to_num)   # converts words to numbers in "experience" column using word2number


# In[20]:


df


# In[25]:


df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].median())

# fills NaN value in "test_score(out of 10)" with median of the datapoints


# In[26]:


df


# In[27]:


reg = linear_model.LinearRegression()


# In[38]:


reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])


# In[33]:


reg.predict([[2,9,6]])  #candidate with 2 years of experience, 9/10 test_score, 6/10 interview_score


# In[34]:


reg.predict([[12,10,10]])   #candidate with 12 years of experience, 10/10 test_score, 10/10 interview_score


# In[ ]:





# In[ ]:




