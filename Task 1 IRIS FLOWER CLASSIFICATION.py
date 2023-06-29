#!/usr/bin/env python
# coding: utf-8

# # Task 1 Oasis Infobytes : Data Science Internship

# In[1]:


# Step 1 : Importing Libraries


# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[3]:


# Step 2 : Importing Dataset


# In[6]:


a=pd.read_csv("C:\\Users\\Rupa\\Documents\\Iris.csv")
a


# In[7]:


# For first 5 rows
a.head()


# In[8]:


# For last 5 rows
a.tail()


# In[9]:


#Dimension of dataset
a.shape 


# In[10]:


# descriptive Statistics
a.describe() 


# In[11]:


# Information about dataset
a.info() 


# In[12]:


# Step 3 : Exploratory Data Analysis


# In[13]:


# Using heatmap to know the correlation among the variables
data = a[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]]
print(data.head())


# In[14]:


sns.heatmap(data.corr(),annot= True)


# In[15]:


a.hist(color="orange",figsize=(9,7))
plt.show()


# In[16]:


# Step 4 : Building ML model for classification


# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


# Dividing dataset into train and test data
X=data.drop('Species',axis = 1)
X


# In[19]:


y=data['Species']
y


# In[20]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[21]:


X_train.head()


# In[22]:


X_test.head()


# In[23]:


y_train.head()


# In[24]:


y_test.head()


# In[25]:


X_train.shape


# In[26]:


X_test.shape


# In[27]:


y_train.shape


# In[28]:


y_test.shape


# In[29]:


model = LogisticRegression()


# In[30]:


model.fit(X_train,y_train)


# In[31]:


model_pred= model.predict(X_test)
model_pred


# In[32]:


from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


# In[33]:


#confusion matrix
confusion_matrix(model_pred,y_test)


# In[34]:


#accuracy
print("Accuracy:", accuracy_score(model_pred,y_test)*100)


# In[35]:


#Checking our predict function is working or not
ans = model.predict([[1.2, 1.5, 1.6, 1.2]])
ans


# In[ ]:




