#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[2]:


heart_data = pd.read_csv('F:\Data science\My practice work\heart.csv')
heart_data


# In[3]:


df = heart_data
df


# In[4]:


df.info()


# In[5]:


df.duplicated().sum()


# In[6]:


df.describe()


# In[7]:


df.dtypes


# In[8]:


plt.figure(figsize = (20,10))
sns.barplot(df)
plt.show()


# In[9]:


sns.boxplot(df)
plt.show()


# In[10]:


df['Sex'].unique()


# In[11]:


df['ChestPainType'].unique()


# In[12]:


df['RestingECG'].unique()


# In[13]:


df['ExerciseAngina'].unique()


# In[14]:


df['ST_Slope'].unique()


# In[15]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for columns in df.columns:
    if df[columns].dtypes != object:
        continue
    df[columns] = le.fit_transform(df[columns])
df


# In[16]:


df.dtypes


# In[17]:


df.isna().sum()


# In[18]:


plt.figure(figsize = (15,8))
sns.countplot(x = 'Cholesterol', hue = 'HeartDisease', data = df)
plt.axis([10,25,0,5])
plt.show()


# In[19]:


plt.figure(figsize = (15,8))
sns.countplot(x = 'Age', hue = 'HeartDisease', data = df)
plt.axis([0,50,0,35])
plt.show()


# In[20]:


plt.figure(figsize = (15,8))
sns.countplot(x = 'Sex', hue = 'HeartDisease', data = df)
plt.show()


# In[21]:


plt.figure(figsize = (15,8))
sns.countplot(x = 'ChestPainType', hue = 'HeartDisease', data = df)
plt.show()


# In[22]:


plt.figure(figsize = (25,10))
sns.countplot(x = 'RestingBP', hue = 'HeartDisease', data = df)
plt.show()


# In[23]:


plt.figure(figsize = (15,8))
sns.countplot(x = 'FastingBS', hue = 'HeartDisease', data = df)
plt.show()


# In[24]:


plt.figure(figsize = (15,8))
sns.countplot(x = 'RestingECG', hue = 'HeartDisease', data = df)
plt.show()


# In[25]:


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 30,15
sns.countplot(x = 'MaxHR', hue = 'HeartDisease', data = df)
plt.show()


# In[26]:


plt.figure(figsize = (15,8))
sns.countplot(x = 'ExerciseAngina', hue = 'HeartDisease', data = df)
plt.show()


# In[27]:


plt.figure(figsize = (15,8))
sns.countplot(x = 'Oldpeak', hue = 'HeartDisease', data = df)
plt.show()


# In[28]:


plt.figure(figsize = (15,8))
sns.countplot(x = 'ST_Slope', hue = 'HeartDisease', data = df)
plt.show()


# In[29]:


df.hist(color = 'c',bins = 20)
plt.show()


# In[30]:


df.corr()


# In[31]:


sns.pairplot(df)


# In[32]:


plt.figure(figsize = (15,8))
sns.heatmap(df.corr(),annot = True)
plt.show()


# In[33]:


x = df.iloc[:,:-1]
x


# In[34]:


y = df.iloc[:,-1]
y


# In[35]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
mms = mms.fit_transform(x)
mms


# In[36]:


print(pd.DataFrame(mms))


# In[37]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(mms,y,test_size = 0.2, random_state = 42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[38]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(x_train,y_train)
lr_pred = LR.predict(x_test)
lr_pred


# In[39]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_score(y_test,lr_pred)


# In[40]:


confusion_matrix(y_test,lr_pred)


# In[41]:


print(classification_report(y_test,lr_pred))


# In[42]:


from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier()
dc.fit(x_train,y_train)
dc_pred = dc.predict(x_test)
dc_pred


# In[43]:


print('Accuracy Score: ',accuracy_score(y_test,dc_pred))
print('-'*50)
print('Confusion Matrix\n', confusion_matrix(y_test,dc_pred))
print('-'*50)
print('Classification Report\n', classification_report(y_test,dc_pred))


# In[44]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)
rf_pred


# In[45]:


print('Accuracy Score: ',accuracy_score(y_test,rf_pred))
print('-'*50)
print('Confusion Matrix\n', confusion_matrix(y_test,rf_pred))
print('-'*50)
print('Classification Report\n', classification_report(y_test,rf_pred))


# In[46]:


from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors = 5)
kn.fit(x_train,y_train)
kn_pred = kn.predict(x_test)
print(kn_pred)


# In[47]:


print('Accuracy Score: ',accuracy_score(y_test,kn_pred))
print('-'*50)
print('Confusion Matrix\n', confusion_matrix(y_test,kn_pred))
print('-'*50)
print('Classification Report\n', classification_report(y_test,kn_pred))


# In[48]:


from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
gnb = GaussianNB()
bnb = BernoulliNB()
mnb = MultinomialNB()
gnb.fit(x_train,y_train)
gnb_pred = gnb.predict(x_test)
print('gnb_pred\n',gnb_pred)
bnb.fit(x_train,y_train)
bnb_pred = bnb.predict(x_test)
print('bnb_pred\n',bnb_pred)


# In[49]:


print('Accuracy Score: ',accuracy_score(y_test,gnb_pred))
print('-'*50)
print('Confusion Matrix\n', confusion_matrix(y_test,gnb_pred))
print('-'*50)
print('Classification Report\n', classification_report(y_test,gnb_pred))
print('Accuracy Score: ',accuracy_score(y_test,bnb_pred))
print('-'*50)
print('Confusion Matrix\n', confusion_matrix(y_test,bnb_pred))
print('-'*50)
print('Classification Report\n', classification_report(y_test,bnb_pred))


# In[50]:


from sklearn.preprocessing import MinMaxScaler
minmax_x = MinMaxScaler()
minmax_x = minmax_x.fit_transform(x)
print(minmax_x)


# In[51]:


x_train,x_test,y_train,y_test = train_test_split(minmax_x,y,test_size = 0.2,random_state = 42)
mnb.fit(x_train,y_train)
mnb_pred = mnb.predict(x_test)
mnb_pred


# In[52]:


print('Accuracy Score: ',accuracy_score(y_test,mnb_pred))
print('-'*50)
print('Confusion Matrix\n', confusion_matrix(y_test,mnb_pred))
print('-'*50)
print('Classification Report\n', classification_report(y_test,mnb_pred))


# In[53]:


x_train,x_test,y_train,y_test = train_test_split(minmax_x,y,test_size = 0.2)


# In[54]:


from sklearn.svm import SVC
svc = SVC(kernel = 'rbf')
svc.fit(x_train,y_train)
svc_pred = svc.predict(x_test)
svc_pred


# In[55]:


print('Accuracy Score: ',accuracy_score(y_test,svc_pred))
print('-'*50)
print('Confusion Matrix\n', confusion_matrix(y_test,svc_pred))
print('-'*50)
print('Classification Report\n', classification_report(y_test,svc_pred))


# In[56]:


from sklearn.svm import SVC
svc = SVC(kernel = 'poly')
svc.fit(x_train,y_train)
svc_pred = svc.predict(x_test)
svc_pred


# In[57]:


print('Accuracy Score: ',accuracy_score(y_test,svc_pred))
print('-'*50)
print('Confusion Matrix\n', confusion_matrix(y_test,svc_pred))
print('-'*50)
print('Classification Report\n', classification_report(y_test,svc_pred))


# In[58]:


from sklearn.svm import SVC
svc = SVC(kernel = 'sigmoid')
svc.fit(x_train,y_train)
svc_pred = svc.predict(x_test)
svc_pred


# In[59]:


print('Accuracy Score: ',accuracy_score(y_test,svc_pred))
print('-'*50)
print('Confusion Matrix\n', confusion_matrix(y_test,svc_pred))
print('-'*50)
print('Classification Report\n', classification_report(y_test,svc_pred))


# In[60]:


from sklearn.svm import SVC
svc = SVC(kernel = 'linear' )
svc.fit(x_train,y_train)
svc_pred = svc.predict(x_test)
svc_pred


# In[61]:


print('Accuracy Score: ',accuracy_score(y_test,svc_pred))
print('-'*50)
print('Confusion Matrix\n', confusion_matrix(y_test,svc_pred))
print('-'*50)
print('Classification Report\n', classification_report(y_test,svc_pred))


# In[62]:


#Random Forest model gives the maximum accuracy followed by KNN and Logistic Regression

