#!/usr/bin/env python
# coding: utf-8

# # Task 2 Oasis Infobytes : Data Science Internship

# In[1]:


# Importing Libraries


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[23]:


a = pd.read_csv("C:\\Users\\Rupa\\Documents\\Unemployment_Rate_upto_11_2020.csv")  


# In[24]:


a.head(10)


# In[25]:


a.tail(10)


# In[26]:


a.shape


# In[27]:


a.info()


# In[28]:


print(a.isnull().sum())


# In[29]:


a.describe()


# In[30]:


# Data Visualization


# In[31]:


plt.figure(figsize=(6, 4))
sns.heatmap(a.corr(),annot=True)
plt.show()


# In[32]:


sns.countplot(x='Region.1',data=a)
plt.ylabel('Estimated Unemployment Rate (%)')


# In[33]:


freq = a['Region.1'].value_counts()
freq


# In[34]:


freq.plot(kind='pie',startangle = 140)
plt.legend()
plt.show()


# In[35]:


# To analyze the data statewise


# In[36]:


a.columns =["States","Date","Frequency","Estimated Unemployment Rate",
              "Estimated Employed","Estimated Labour Participation Rate",
              "Region","longitude","latitude"]
print(a)


# In[37]:


unemployment_data = a[["States","Region","Estimated Unemployment Rate"]]
figure = px.sunburst(unemployment_data,path=["Region","States"],
                    values="Estimated Unemployment Rate",
                    width=700,height=700,
                    title="Unemployment rate in India during Covid-19")
figure.show()


# In[ ]:




