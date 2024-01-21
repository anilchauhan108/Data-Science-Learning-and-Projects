#!/usr/bin/env python
# coding: utf-8

# In[96]:


import pandas as pd
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt


# In[97]:


df = pd.read_csv('C:\DS practice\Iris.csv')


# In[98]:


dir(df)


# In[99]:


df.head(10)


# In[115]:


df.shape


# In[100]:


df.Species.unique()


# In[101]:


from sklearn.model_selection import train_test_split


# In[102]:


X_train, X_test, y_train, y_test = train_test_split(df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']],
                                                    df['Species'], train_size = 0.85)


# In[103]:


from sklearn.linear_model import LogisticRegression


# In[104]:


model = LogisticRegression()


# In[105]:


model.fit(X_train,y_train)


# In[106]:


model.predict([[10,1,1,0.3]])


# In[107]:


model.score(X_train,y_train)


# In[108]:


y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)
cm


# In[109]:


plt.figure(figsize = (5,4))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[110]:


model.predict([[10,1,1,0.3]])


# In[111]:


model.predict([[10,5,1,0.3]])


# In[112]:


model.predict([[10,6,5,1]])


# In[113]:


model.predict([[5,4,5,2]])


# In[ ]:





# In[ ]:




