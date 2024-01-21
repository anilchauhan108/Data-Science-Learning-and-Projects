#!/usr/bin/env python
# coding: utf-8

# In[221]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[222]:


df = pd.read_csv('C:\DS practice\HR_comma_sep.csv')


# In[223]:


df.head(10)


# In[224]:


df.Department.unique()


# In[225]:


left = df[df.left == 1]


# In[226]:


retained = df[df.left ==0]


# In[227]:


df.groupby('left').mean(numeric_only=True)  #Deprecated since version 1.5.0: Specifying numeric_only=None is deprecated. 
                                            #The default value will be False in a future version of pandas.
                                            #Changed default of numeric_only in various DataFrameGroupBy methods; all methods now default to numeric_only=False


# **From here we can see that employees who left the organisation had
# i)   Low satisfaction_level
# ii)  More average_monthly_hours
# iii) Less promotions**

# In[228]:


pd.crosstab(df.salary,df.left).plot(kind='bar')   #Huge salary gap amoungst the employees who left the org vs who were retained


# In[229]:


pd.crosstab(df.Department,df.left).plot(kind='bar')


# **From above Data exploration we can conclude that following independent variable will have major impact on our model.
# i)   Satisfaction level
# ii)  Average_monthly_hours
# iii) Less promotions
# iv)  Salary**

# In[230]:


from sklearn.preprocessing import LabelEncoder


# In[231]:


le = LabelEncoder()


# In[232]:


dfle = df
dfle.salary = le.fit_transform(dfle.salary)


# In[233]:


from sklearn.model_selection import train_test_split


# In[234]:


X_train,X_test,y_train,y_test = train_test_split(dfle[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years'
                                                    ,'salary']],dfle.left, train_size = 0.7)


# In[235]:


X_train


# In[236]:


X_test


# In[237]:


from sklearn.linear_model import LogisticRegression


# In[238]:


lr = LogisticRegression()


# In[239]:


lr.fit(X_train,y_train)


# In[240]:


lr.predict(X_test)


# In[241]:


lr.score(X_train,y_train)


# In[242]:


pd.crosstab(dfle.salary,dfle.left).plot(kind = "bar")


# In[243]:


pd.crosstab(df.salary,df.left).plot(kind = "bar")


# In[ ]:




