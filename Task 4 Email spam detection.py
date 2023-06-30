#!/usr/bin/env python
# coding: utf-8

# # Task 4 Email spam detection

# In[1]:


# Step 1 : Data loading


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


a = pd.read_csv("C:\\Users\\Rupa\\Documents\\spam.csv",encoding="latin-1")  
a


# In[4]:


# Step 2 : Data Cleaning


# In[5]:


a.columns


# In[6]:


a.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)


# In[7]:


a.head(25)


# In[8]:


a.rename(columns={'v1':'target','v2':'mails'},inplace =True)
a


# In[9]:


a.isnull().sum()


# In[10]:


a.info()


# In[11]:


a.shape


# In[12]:


# To Check duplicate values in dataset
a.duplicated().sum()


# In[13]:


a =a.drop_duplicates(keep='first')


# In[14]:


a.duplicated().sum()


# In[15]:


#Step 3 : Exploratory data analysis


# In[16]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[17]:


a['target'] = encoder.fit_transform(a['target'])


# In[18]:


a.head()


# In[19]:


a['target'].value_counts()


# In[20]:


plt.pie(a['target'].value_counts(), labels = ['ham','spam'],autopct="%0.2f")
plt.show()


# In[21]:


#Step 4 : Performing Logistic Regression


# In[22]:


X = a['mails']
y = a['target']


# In[23]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state = 3)


# In[24]:


feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase ='True')


# In[25]:


print(X)


# In[26]:


y_train = y_train.astype('int')
y_test = y_test.astype('int')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


# In[27]:


print(X_train)


# In[28]:


print(X_train_features)


# In[29]:


# Training the Ml Logistic regression model
model = LogisticRegression()
model.fit(X_train_features, y_train)


# In[32]:


predict_train_data = model.predict(X_train_features)
accuracy_train_data = accuracy_score(y_train, predict_train_data)
print('accuracy_train_data:', accuracy_train_data)


# In[31]:


predict_test_data = model.predict(X_test_features)
accuracy_test_data = accuracy_score(y_test, predict_test_data)
print('accuracy_test_data:', accuracy_test_data)


# # The accuracy for test data is 96%

# In[ ]:




