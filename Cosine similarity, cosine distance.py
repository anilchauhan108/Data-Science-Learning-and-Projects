#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


# In[5]:


cosine_similarity([[5,1]],[[10,2]])


# In[6]:


cosine_distances([[5,1]],[[10,2]])


# In[15]:


cosine_similarity([[4,3]],[[1,4]])


# In[16]:


cosine_distances([[4,3]],[[1,4]])


# In[ ]:


# this can also be done using tensorflow

