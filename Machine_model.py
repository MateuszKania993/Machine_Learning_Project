#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('skychallenge_clean_data.csv')


df = df.head(10000)
# df = df.drop(columns = ['condition', 'fuel', 'transmission'])
df = df.drop_duplicates()
df


# In[2]:


from statsmodels.graphics.correlation import plot_corr 
from sklearn.model_selection import train_test_split
import seaborn as sns
plt.style.use('seaborn') 

x = df.drop('price', axis=1)
y = df[['price']]

seed = 42

test_data_size = 0.25

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = test_data_size, random_state = seed)

train_data = pd.concat([X_train, Y_train], axis = 1)
test_data = pd.concat([X_test, Y_test], axis = 1)

corrMatrix = train_data.corr(method= 'pearson')
xnames = list(train_data.columns)
ynames = list(train_data.columns)


# In[ ]:


from sklearn import linear_model
from sklearn.metrics import explained_variance_score, accuracy_score
from sklearn.svm import SVC


X_train = np.array(X_train)
Y_train = np.ravel(np.array(Y_train))
X_test = np.array(X_test)
Y_test = np.array(Y_test)

clf = SVC()

clf.fit(X_train, Y_train)

preds = clf.predict(X_train)

# print(type(preds))

# print(type(Y_test))

train_acc = explained_variance_score(Y_train, preds)


# In[24]:


print(train_acc)


# In[9]:


from sklearn import linear_model
from sklearn.metrics import explained_variance_score, accuracy_score


X_train = np.array(X_train)
Y_train = np.ravel(np.array(Y_train))
X_test = np.array(X_test)
Y_test = np.array(Y_test)


clf = linear_model.SGDClassifier(max_iter = 10000)

clf.fit(X_train, Y_train)

preds = clf.predict(X_test)


train_acc = accuracy_score(Y_test, preds)

print(train_acc)


# In[3]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=10, min_samples_leaf=3)
rf_model.fit(X_train, Y_train)

train_preds = rf_model.predict(X_train)

train_acc = accuracy_score(Y_train, train_preds)
print(train_acc)


# In[ ]:




