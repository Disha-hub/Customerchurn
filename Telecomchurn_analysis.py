#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import  Image
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.ticker as mtick
import seaborn as sns
sns.set(style='white')

import io
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import roc_curve , auc
from sklearn.svm import SVC


# In[2]:


tc=pd.read_csv("Telecomchurn.csv")


# In[3]:


tc.head()


# In[4]:


tc.describe()


# In[5]:


tc.columns.values


# In[6]:


tc.dtypes


# In[7]:


tc.TotalCharges=pd.to_numeric(tc.TotalCharges, errors='coerce')
tc.dtypes


# In[8]:


tc.isnull().sum()


# In[9]:


#lets remove the missing value
tc.dropna(inplace=True)
tc.isnull().sum()


# In[10]:


tc=tc.reset_index()[tc.columns]


# In[11]:


#Replacing columns and cateogarizing tenure
cr = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
for i in cr : 
    tc[i]  = tc[i].replace({'No internet service' : 'No'})

tc["SeniorCitizen"] = tc["SeniorCitizen"].replace({1:"Yes",0:"No"})
tc["MultipleLines"] = tc["MultipleLines"].replace({"No phone service":"No"})


def tenure_lab(tc) :
#changing conntinuous variable to cateogorical variable(tenure)   
    if tc["tenure"] <= 12 :
        return "Tenure_0-12"
    elif (tc["tenure"] > 12) & (tc["tenure"] <= 24 ):
        return "Tenure_12-24"
    elif (tc["tenure"] > 24) & (tc["tenure"] <= 48) :
        return "Tenure_24-48"
    elif (tc["tenure"] > 48) & (tc["tenure"] <= 60) :
        return "Tenure_48-60"
    elif tc["tenure"] > 60 :
        return "Tenure_gt_60"
tc["tenure_group"] = tc.apply(lambda tc:tenure_lab(tc),
                                    axis = 1)


# In[12]:


tc['Churn'].replace(to_replace='Yes', value=1, inplace=True)
tc['Churn'].replace(to_replace='No', value=0, inplace=True)
tc=tc.iloc[:,1:]
tc.drop('tenure', axis=1, inplace=True)
tc.head()


# In[13]:


#correlation
tc_dummies=pd.get_dummies(tc)
plt.figure(figsize=(15,10))
tc_dummies.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')


# In[14]:


#Lets see out Target column churn
ax = sns.catplot(y="Churn", kind="count", data=tc, height=2.6, aspect=2.5, orient='h')


# In[15]:


#Histogram of Numerical variables
tc.hist(bins=20,figsize=(20,15))
plt.show()


# In[16]:


#tenure group vs Churn
fig = plt.gcf()
fig.set_size_inches( 7, 5)
plt.title('Churn by tenure_group')
sns.countplot(tc['tenure_group'],hue=tc['Churn'])


# In[17]:


fig = plt.gcf()
fig.set_size_inches( 7, 5)
plt.title('Churn by Contract')
sns.countplot(tc['Contract'],hue=tc['Churn'])


# In[18]:


fig = plt.gcf()
fig.set_size_inches( 7, 5)
plt.title('Churn by PaymentMethod')
sns.countplot(tc['PaymentMethod'], hue=tc['Churn'])


# In[19]:


fig = plt.gcf()
fig.set_size_inches( 7, 5)
plt.title('Churn by InternetService')
sns.countplot(tc['InternetService'], hue=tc['Churn'])


# In[20]:


categorical_var=[i for i in tc.columns if tc[i].dtypes=='object']
categorical_var_Nochurn=categorical_var[:-1]
fig , ax = plt.subplots(4,4,figsize=(20,20))
for axi , var in zip(ax.flat,categorical_var_Nochurn):
    sns.countplot(x=tc.Churn,hue=tc[var],ax=axi)


# In[21]:


tc.head()


# In[22]:


#Data Normalization
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
for x in [i for i in tc.columns if len(tc[i].unique())==2]:
    print(x, tc[x].unique())
    tc[x]= label_encoder.fit_transform(tc[x])


# In[23]:


[[x, tc[x].unique()] for x in [i for i in tc.columns if len(tc[i].unique())<10]]


# In[24]:


tc= pd.get_dummies(tc, columns= [i for i in tc.columns if tc[i].dtypes=='object'],drop_first=True)


# In[25]:


[[x, tc[x].unique()] for x in [i for i in tc.columns if len(tc[i].unique())<10]]


# In[26]:


#Data splitting
trgt=tc['Churn'].values
attr=tc.drop('Churn', axis=1)
from sklearn.model_selection import train_test_split
attr_train, attr_test, trgt_train, trgt_test=train_test_split(attr, trgt, test_size=0.2)


# In[27]:


#standardisation
sc= StandardScaler()
attr_train = sc.fit_transform(attr_train)
attr_train=pd.DataFrame(attr_train,columns=attr.columns)
attr_test=sc.transform(attr_test)


# In[28]:


#Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
lr=LogisticRegression()
result=lr.fit(attr_train, trgt_train)
prediction_test=lr.predict(attr_test)
print(metrics.accuracy_score(trgt_test,prediction_test))
logit  = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)


# In[29]:


#Confusion Matrix with heatmap
cnfsn_matrix=confusion_matrix( trgt_test ,prediction_test)
ax=plt.subplot()
sns.heatmap(cnfsn_matrix, annot=True, fmt="d",xticklabels=['Yes','No'],
            yticklabels=['yes','No'],cbar=False,ax=ax)
plt.ylabel('True Lable')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix")


# In[30]:


print(classification_report(trgt_test, prediction_test, target_names=['chrun_yes','churn_no']))


# In[31]:


weights=pd.Series(lr.coef_[0],index=attr.columns.values)
#Factros that are increasing churn rate 
print(weights.sort_values(ascending=False)[:10].plot(kind='bar'))


# In[32]:


#Factors that can reduce churn Rate
print(weights.sort_values(ascending=False)[-10:].plot(kind='bar'))


# In[33]:


#ROC curve
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve

trgt_pred_prob=lr.predict_proba(attr_test)[:,1]
fpr,tpr,threshold=roc_curve(trgt_test,trgt_pred_prob)
roc_auc = metrics.auc(fpr, tpr)
plt.plot([0,1],[0,1],'k--', label = 'AUC = %0.2f' % roc_auc )
plt.plot(fpr,tpr,label='Logistic regression')
plt.xlabel('False Positve Rate')
plt.ylabel('True Psoitive Rate')
plt.title('ROC curve for Logistic Regression')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()


# In[34]:


#Support Vector Machine
from sklearn.svm import SVC
svmc= SVC(probability=True)
svmc.fit(attr_train,trgt_train)
svm_pred=svmc.predict(attr_test)
print(metrics.accuracy_score(trgt_test,svm_pred))


# In[35]:


print(classification_report(trgt_test, svm_pred, target_names=['chrun_yes','churn_no']))


# In[36]:


cnfsn_matrix=confusion_matrix( trgt_test ,svm_pred)
ax=plt.subplot()
sns.heatmap(cnfsn_matrix, annot=True, fmt="d",xticklabels=['Yes','No'],
            yticklabels=['yes','No'],cbar=False,ax=ax)
plt.ylabel('True Lable')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix")


# In[37]:


trgt_pred_svm=svmc.predict_proba(attr_test)[:,1]
fpr_svm,tpr_svm,threshold=roc_curve(trgt_test,trgt_pred_svm)
roc_auc = auc(fpr_svm, tpr_svm)
plt.plot([0,1],[0,1],'k--', label = 'Random' )
plt.plot(fpr,tpr,label='ROC curve(area=%0.2f)'%roc_auc)
plt.xlabel('False Positve Rate')
plt.ylabel('True Psoitive Rate')
plt.title('ROC curve for Support vector Machine')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()


# In[43]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
rfc=RandomForestClassifier()
rfc.fit(attr_train, trgt_train)
pred_rfc=rfc.predict(attr_test)
print(metrics.accuracy_score(trgt_test, pred_rfc))


# In[44]:


print(classification_report(trgt_test, pred_rfc, target_names=['chrun_yes','churn_no']))


# In[46]:


cnfsn_matrix=confusion_matrix( trgt_test ,pred_rfc)
ax=plt.subplot()
sns.heatmap(cnfsn_matrix, annot=True, fmt="d",xticklabels=['Yes','No'],
            yticklabels=['yes','No'],cbar=False,ax=ax)
plt.ylabel('True Lable')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix")


# In[48]:


trgt_pred_rfc=rfc.predict_proba(attr_test)[:,1]
fpr_rfc,tpr_rfc,threshold=roc_curve(trgt_test,pred_rfc)
roc_auc = auc(fpr_rfc, tpr_rfc)
plt.plot([0,1],[0,1],'k--', label = 'Random' )
plt.plot(fpr,tpr,label='ROC curve(area=%0.2f)'%roc_auc)
plt.xlabel('False Positve Rate')
plt.ylabel('True Psoitive Rate')
plt.title('ROC curve for Random Forest')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()


# In[49]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(attr_train, trgt_train)
pred_knn=knn.predict(attr_test)
print(metrics.accuracy_score(trgt_test,svm_pred))


# In[50]:


print(classification_report(trgt_test, pred_knn, target_names=['chrun_yes','churn_no']))


# In[51]:


cnfsn_matrix=confusion_matrix( trgt_test ,pred_knn)
ax=plt.subplot()
sns.heatmap(cnfsn_matrix, annot=True, fmt="d",xticklabels=['Yes','No'],
            yticklabels=['yes','No'],cbar=False,ax=ax)
plt.ylabel('True Lable')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix")


# In[52]:


trgt_pred_knn=knn.predict_proba(attr_test)[:,1]
fpr_knn,tpr_knn,threshold=roc_curve(trgt_test,knn_pred)
roc_auc = auc(fpr_knn, tpr_knn)
plt.plot([0,1],[0,1],'k--', label = 'Random' )
plt.plot(fpr_knn,tpr_knn,label='ROC curve(area=%0.2f)'%roc_auc)
plt.xlabel('False Positve Rate')
plt.ylabel('True Psoitive Rate')
plt.title('ROC curve for KNN')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()


# In[53]:


#Model comparision
from sklearn import model_selection

models=[]
models.append(('lr', LogisticRegression()))
models.append(('Rfc', RandomForestClassifier()))
models.append(('svm', SVC()))
models.append(('knn', KNeighborsClassifier()))
results=[]
names=[]
scoring='accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state='seed')
    cv_results = model_selection.cross_val_score(model, attr, trgt, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[55]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[ ]:




