#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
diabetes = pd.read_csv('diabetes.csv')
diabetes.columns 


# In[2]:


diabetes.head()


# In[3]:


print(diabetes['Glucose'].min())


# In[4]:


print("Diabetes data set dimensions : {}".format(diabetes.shape))


# In[5]:


diabetes.groupby('Outcome').size()


# In[6]:


diabetes.groupby('Outcome').hist(figsize=(9, 9))


# In[7]:


a=diabetes.hist(column='Age')


# In[ ]:





# In[8]:


ax = diabetes.hist(column='SkinThickness', bins=25, grid=False, figsize=(12,8), color='#ffbfbf', zorder=2, rwidth=0.9)

ax = ax[0]
for x in ax:

    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)

    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#0d0d00', zorder=1)

    # Remove title
    x.set_title("")

    # Set x-axis label
    x.set_xlabel("Thickness of Skin(mm)", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x.set_ylabel("Number", labelpad=20, weight='bold', size=12)
    


# In[9]:


diabetes.isnull().sum()
diabetes.isna().sum()


# In[10]:


print("Total : ", diabetes[diabetes.BloodPressure == 0].shape[0])


# In[11]:


print(diabetes[diabetes.BloodPressure == 0].groupby('Outcome')['Age'].count())


# In[12]:


diabetes_mod = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]
print(diabetes_mod.shape)
print(diabetes_mod['DiabetesPedigreeFunction'].min())
print(diabetes_mod['DiabetesPedigreeFunction'].max())


# In[13]:


feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_mod[feature_names]
y = diabetes_mod.Outcome


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[15]:


models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC(gamma='auto')))
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_mod.Outcome, random_state=0)


# In[17]:


names = []
scores = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
    
    #graph
    yes=no=0
    yes=int(accuracy_score(y_test, y_pred)*181)
    no=int(181-yes)
    print(name,yes,no)
    data = [['Correctly Predicted', yes], ['Falsely PRedicted', no]] 
    df1 = pd.DataFrame(data, columns = ['Predictions', 'Number']) 
#     print(df1)
    var = df1.groupby('Predictions').Number.sum() 
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('Predictions')
    ax1.set_ylabel('Number of Predictions')
    ax1.set_title(name)
    var.plot(kind='bar')
    
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)

# plot
axis = sns.barplot(x = 'Name', y = 'Score', data = tr_split)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()


# In[18]:



data = [[1,85,66,29,0,26.6,0.351,31]] 
df4 = pd.DataFrame(data, columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']) 
model = KNeighborsClassifier()
model.fit(X_test,y_test)
output=model.predict(data)
print("KNN->",output)


# In[19]:


from sklearn.model_selection import KFold 
names = []
scores = []
models1 = []
models1.append(('KNN', KNeighborsClassifier()))
models1.append(('SVC', SVC(gamma='auto')))
models1.append(('DT', DecisionTreeClassifier()))

for name, model in models1:
    
    kfold = KFold(n_splits=10, random_state=10) 
    score = cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean()
    
    names.append(name)
    scores.append(score)
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)


# In[20]:


axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()


# In[ ]:




