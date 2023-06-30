#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction

# In[6]:


#import all libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[7]:


#Step 2 Importing Data set


# In[8]:


a = pd.read_csv("C:\\Users\\Rupa\\Documents\\car.csv")  
a


# In[9]:


a.tail()


# In[10]:


a.shape


# In[11]:


a.info()


# In[14]:


a.describe()


# In[13]:


#Step 3 : Exploratory Data Analysis


# In[15]:


a.duplicated(subset = ['car_ID']).sum()


# In[16]:


a = a.drop(['car_ID'], axis =1)
a.isnull().sum()


# In[17]:


a['symboling'].value_counts()


# In[18]:


a['CarName'].value_counts()


# In[19]:


a['car_company'] = a['CarName'].apply(lambda x:x.split(' ')[0])
a.head()


# In[20]:


a = a.drop(['CarName'], axis =1)
a['car_company'].value_counts()


# In[21]:


a['car_company'].replace('toyouta', 'toyota',inplace=True)
a['car_company'].replace('Nissan', 'nissan',inplace=True)
a['car_company'].replace('maxda', 'mazda',inplace=True)
a['car_company'].replace('vokswagen', 'volkswagen',inplace=True)
a['car_company'].replace('vw', 'volkswagen',inplace=True)
a['car_company'].replace('porcshce', 'porsche',inplace=True)


# In[22]:


a['car_company'].value_counts()


# In[23]:


a['fueltype'].value_counts()


# In[24]:


a['aspiration'].value_counts()


# In[25]:


a['doornumber'].value_counts()


# In[26]:


def number_(x):
    return x.map({'four':4, 'two': 2})
    
a['doornumber'] = a[['doornumber']].apply(number_)


# In[27]:


a['doornumber'].value_counts()


# In[28]:


sns.distplot(a['wheelbase'],color='red')
plt.show()


# In[29]:


a['carlength'].value_counts().head()


# In[30]:


sns.distplot(a['carlength'],color='green')
plt.show()


# In[31]:


def convert_number(x):
    return x.map({'two':2, 'three':3, 'four':4,'five':5, 'six':6,'eight':8,'twelve':12})

a['cylindernumber'] = a[['cylindernumber']].apply(convert_number)


# In[32]:


a['cylindernumber'].value_counts()


# In[33]:


b =a.select_dtypes(include =['int64','float64'])
b.head()


# In[34]:


plt.figure(figsize = (30,30))
sns.pairplot(b)
plt.show()


# In[35]:


plt.figure(figsize = (20,20))
sns.heatmap(a.corr(), annot = True)
plt.show()


# In[36]:


categorical_cols = a.select_dtypes(include = ['object'])
categorical_cols.head()


# In[37]:


plt.figure(figsize = (20,12))
plt.subplot(3,3,1)
sns.boxplot(x = 'fueltype', y = 'price', data = a)
plt.subplot(3,3,2)
sns.boxplot(x = 'aspiration', y = 'price', data = a)
plt.subplot(3,3,3)
sns.boxplot(x = 'carbody', y = 'price', data = a)
plt.subplot(3,3,4)
sns.boxplot(x = 'drivewheel', y = 'price', data = a)
plt.subplot(3,3,5)
sns.boxplot(x = 'enginelocation', y = 'price', data = a)
plt.subplot(3,3,6)
sns.boxplot(x = 'enginetype', y = 'price', data = a)
plt.subplot(3,3,7)
sns.boxplot(x = 'fuelsystem', y = 'price', data = a)


# In[38]:


cars_dummies = pd.get_dummies(categorical_cols, drop_first = True)
cars_dummies.head()


# In[39]:


car_df  = pd.concat([a, cars_dummies], axis =1)
car_df = car_df.drop(['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation',
       'enginetype', 'fuelsystem', 'car_company'], axis =1)


# In[40]:


#Step 4 : Building ML model


# In[41]:


df_train, df_test = train_test_split(car_df, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[42]:


df_train.shape


# In[43]:


df_test.shape


# In[44]:


b.columns


# In[45]:


col_list = ['symboling', 'doornumber', 'wheelbase', 'carlength', 'carwidth','carheight', 'curbweight', 'cylindernumber', 'enginesize', 'boreratio',
            'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']
scaler = StandardScaler()
df_train[col_list] = scaler.fit_transform(df_train[col_list])


# In[46]:


df_train.describe()


# In[47]:


y_train = df_train.pop('price')
X_train = df_train


# In[48]:


lr = LinearRegression()
lr.fit(X_train,y_train)


# In[49]:


rfe = RFE(lr,step=15)
rfe.fit(X_train, y_train)


# In[50]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[51]:


cols = X_train.columns[rfe.support_]
cols


# In[52]:


#Model 1:


# In[53]:


X1 = X_train[cols]
X1_sm = sm.add_constant(X1)

lr_1 = sm.OLS(y_train,X1_sm).fit()
print(lr_1.summary())


# In[55]:


vif = pd.DataFrame()
vif['Features'] = X1.columns
vif['VIF'] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[56]:


lr2 = LinearRegression()

rfe2 = RFE(lr2,step=10)
rfe2.fit(X_train,y_train)


# In[57]:


list(zip(X_train.columns,rfe2.support_,rfe2.ranking_))


# In[58]:


supported_cols = X_train.columns[rfe2.support_]
supported_cols 


# In[59]:


#Model 2:


# In[60]:


X2 = X_train[supported_cols]
X2_sm = sm.add_constant(X2)

model_2 = sm.OLS(y_train,X2_sm).fit()
print(model_2.summary())


# In[61]:


vif = pd.DataFrame()
vif['Features'] = X2.columns
vif['VIF'] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[62]:


#Model 3:


# In[63]:


X3 = X2.drop(['car_company_subaru'], axis =1)
X3_sm = sm.add_constant(X3)

Model_3 = sm.OLS(y_train,X3_sm).fit()
print(Model_3.summary())


# In[64]:


vif = pd.DataFrame()
vif['Features'] = X3.columns
vif['VIF'] = [variance_inflation_factor(X3.values, i) for i in range(X3.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[65]:


#Model 4:


# In[66]:


X4 = X3.drop(['enginetype_ohcf'], axis =1)
X4_sm = sm.add_constant(X4)

Model_4 = sm.OLS(y_train,X4_sm).fit()
print(Model_4.summary())


# In[67]:


vif = pd.DataFrame()
vif['Features'] = X4.columns
vif['VIF'] = [variance_inflation_factor(X4.values, i) for i in range(X4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[68]:


#Model 5:


# In[70]:


X5 = X4.drop(['car_company_peugeot'], axis =1)
X5_sm = sm.add_constant(X5)

Model_5 = sm.OLS(y_train,X5_sm).fit()
print(Model_5.summary())


# In[71]:


vif = pd.DataFrame()
vif['Features'] = X5.columns
vif['VIF'] = [variance_inflation_factor(X5.values, i) for i in range(X5.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[72]:


#Model 6:


# In[73]:


X6 = X5.drop(['enginetype_l'], axis =1)
X6_sm = sm.add_constant(X6)

Model_6 = sm.OLS(y_train,X6_sm).fit()
print(Model_6.summary())


# In[74]:


vif = pd.DataFrame()
vif['Features'] = X6.columns
vif['VIF'] = [variance_inflation_factor(X6.values, i) for i in range(X6.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[75]:


y_train_pred = Model_6.predict(X6_sm)
y_train_pred.head()


# In[76]:


Residual = y_train- y_train_pred


# In[77]:


final_cols = X6.columns


# In[81]:


X_train_model6= X_train[final_cols]
X_train_model6.head()


# In[82]:


X_train_sm = sm.add_constant(X_train_model6)


# In[84]:


y_pred = Model_6.predict(X_train_sm)


# In[85]:


y_pred.head()


# In[93]:


plt.scatter(y_train, y_pred)
plt.xlabel('y_train')
plt.ylabel('y_pred')
plt.show()


# In[94]:


r_squ = r2_score(y_train,y_pred)
r_squ


# In[ ]:




