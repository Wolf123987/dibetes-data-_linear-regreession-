#!/usr/bin/env python
# coding: utf-8

# In[46]:


## importing the required libraries 
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import  mean_squared_error
from sklearn import datasets 
diabetes=datasets.load_diabetes() ## importing the sklearn sample data called as diabetes
ram=diabetes.data   ### taking impit label as ram 
print(); print(ram.shape);
sita=diabetes.data ## taking target as sita 
print();print(sita.shape); ## getting the shape of data 


# In[12]:


diabetes_ram=diabetes.data[:,np.newaxis,2]


# In[60]:


print(" diabetes_ram_data is ",diabetes_ram_test)


# In[63]:


print(" diabetes_sita_data is ",diabetes_sita_test)


# In[14]:


diabetes_ram_train=diabetes_ram[:-30] ## selecting first 30 data of ram as test 
diabetes_ram_test=diabetes_ram[-30:] ### selecting last 30 data of ram as test


# In[58]:


diabestes_sita_train=diabetes.target[:-30]
diabetes_sita_test=diabetes.target[-30:]
model=linear_model.LinearRegression() ## calling the linear regression function 
model.fit(diabetes_ram_train,diabestes_sita_train) ## fitting the model
diabetes_sita_predict= model.predict(diabetes_ram_test) ## predicting the trget 


# In[59]:


diabetes_sita_predict


# In[29]:


## getting the mean squared error as output 
print ("mean_squared_error is ",mean_squared_error(diabetes_sita_test,diabetes_sita_predict))


# In[49]:


plt.scatter(diabetes_ram_test,diabetes_sita_test)
plt.plot(diabetes_ram_test,diabetes_sita_predict,c="r")


# In[56]:


print("intercenpt is :",model.intercept_)


# In[57]:


print("weight of model is ",model.coef_)


# In[68]:


print ((.0164281 *941.43097333)+153.39713623331644) ## predicted value at 0.0164281


# In[62]:


output


# In[ ]:




