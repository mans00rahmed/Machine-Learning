#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[43]:


df = pd.read_csv("prices.csv")
df


# In[44]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df.area,df.prices,color='red',marker='+')


# In[46]:


reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.prices)


# In[55]:


reg.predict([[3300]])

