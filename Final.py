#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('heart.csv', header=0)
df.head()


# In[3]:


df.shape


# # Data visualization

# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


sns.countplot(x='HeartDisease',data=df,palette="muted")
plt.title("Disease distribution")
plt.xlabel("0:Doesn't have disease   1:Has disease  ")


# In[6]:


fig,axes = plt.subplots(1,2)
ax = df.HeartDisease.value_counts().plot(kind="bar",ax=axes[0])
ax.set_title("Disease distribution")
ax.set_label("1:has disease 0:doesn't have disease")

df.HeartDisease.value_counts().plot(kind="pie",autopct="%.2f%%",labels=['has disease','does not have disead'],ax=axes[1])


# In[7]:


sns.countplot(x='Sex',data=df,palette="Set3")
plt.title("Disease gender distribution")
plt.xlabel("M:male F:female")


# In[8]:


ax1 = plt.subplot(121)
ax = sns.countplot(x = "Sex",hue = 'HeartDisease',data = df,ax = ax1)
ax.set_xlabel("M = Male, F = Female")

ax2 = plt.subplot(222)
df[df['HeartDisease'] == 0].Sex.value_counts().plot(kind="pie",autopct="%.2f%%",labels=['male','female'],ax=ax2)
ax2.set_title("Sex ratio without disease")

ax2 = plt.subplot(224)
df[df['HeartDisease'] == 1].Sex.value_counts().plot(kind="pie",autopct="%.2f%%",labels=['male','female'],ax=ax2)
ax2.set_title("Diseased sex ratio")


# In[9]:


fig,axes = plt.subplots(2,1,figsize=(20,10))
sns.countplot(x="Age",hue="HeartDisease",data=df,ax=axes[0])


age_type = pd.cut(df.Age,bins=[0,45,60,100],include_lowest=True,right=False,labels=['people whose age is 0-45','people whose age is 45-60','people whose age is 60-100'])
age_HeartDisease_df = pd.concat([age_type,df.HeartDisease],axis=1)
sns.countplot(x="Age",hue='HeartDisease',data=age_HeartDisease_df)


# In[10]:


plt.figure(figsize=(8,5))
sns.heatmap(df.corr(),annot=True)


# In[11]:


features = df.drop(columns=['HeartDisease'])
targets = df['HeartDisease']


# In[12]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

features = pd.get_dummies(features)
features_temp = StandardScaler().fit_transform(features)

X_train,X_test,y_train,y_test = train_test_split(features_temp,targets,test_size=0.25,random_state = 0)


# In[13]:


X_train.shape[0]


# In[14]:


X_test.shape[0]


# In[15]:


features


# In[16]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
def plot_cnf_matirx(cnf_matrix,description):
    class_names = [0,1]
    fig,ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks,class_names)
    plt.yticks(tick_marks,class_names)
import pickle


# # Model 1: Logistic regression model and confusion matrix

# In[17]:


from sklearn.linear_model import LogisticRegression 

log_reg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
log_reg.fit(X_train,y_train)


# In[18]:


y_predict_log = log_reg.predict(X_test)
cnf_matrix = confusion_matrix(y_test,y_predict_log)
cnf_matrix


# In[19]:


sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'OrRd',fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('confusion matrix for logistic model')
plt.ylabel('actual value 0/1',fontsize=12)
plt.xlabel('predict value 0/1',fontsize=12)
plt.show()


# In[20]:


Classification_accuracy = accuracy_score(y_test,y_predict_log)
print("Classification accuracy for logistic regression is ",Classification_accuracy,'and error rate is',1-Classification_accuracy)


# In[21]:


pickle.dump(log_reg, open('LG.pkl','wb'))


# # Model 2 : Decision tree and confusion matrix

# In[22]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=10)
tree.fit(X_train,y_train)


# In[23]:


from sklearn.tree import export_graphviz
export_graphviz(tree, 'tree.dot',feature_names = features.columns.values)

get_ipython().system(' dot -Tpng tree.dot -o tree.png')

import matplotlib.pyplot as plt
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')
img = cv2.imread('tree.png')
plt.figure(figsize = (20, 20))
plt.imshow(img)


# In[24]:


y_predict_tree = tree.predict(X_test)
tree_matrix = confusion_matrix(y_test,y_predict_tree)
tree_matrix


# In[25]:


sns.heatmap(pd.DataFrame(tree_matrix), annot = True, cmap = 'OrRd',fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('confusion matrix for decision tree')
plt.ylabel('actual value 0/1',fontsize=12)
plt.xlabel('predict value 0/1',fontsize=12)
plt.show()    


# In[26]:


Classification_accuracy_tree = accuracy_score(y_test,y_predict_tree)
print("Classification accuracy for decision tree model is ",Classification_accuracy_tree,'and error rate is',1-Classification_accuracy_tree)


# In[27]:


pickle.dump(tree, open('DT.pkl','wb'))


# # Model 3 : SVM and confusion matrix

# In[28]:


from sklearn.svm import SVC
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train,y_train)


# In[29]:


y_pred_SVM = svclassifier.predict(X_test)
SVM_matrix = confusion_matrix(y_test,y_pred_SVM)
SVM_matrix


# In[30]:


sns.heatmap(pd.DataFrame(SVM_matrix), annot = True, cmap = 'OrRd',fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('confusion matrix for SVM')
plt.ylabel('actual value 0/1',fontsize=12)
plt.xlabel('predict value 0/1',fontsize=12)
plt.show()    


# In[31]:


Classification_accuracy_SVM = accuracy_score(y_test,y_pred_SVM)
print("Classification accuracy for SVM model is ",Classification_accuracy_SVM,'and error rate is',1-Classification_accuracy_SVM)


# In[32]:


pickle.dump(svclassifier, open('SVM.pkl','wb'))


# # Model 4 : Neural network and confusion matrix

# In[33]:


from keras.models import Sequential
from keras.layers import Dense
neural = Sequential()
neural.add(Dense(12, input_dim = 20, activation='relu'))
neural.add(Dense(10, activation='relu'))
neural.add(Dense(8, activation='relu'))
neural.add(Dense(5, activation='relu'))
neural.add(Dense(1, activation='sigmoid'))
neural.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
neural.summary()


# In[34]:


history = neural.fit(X_train, y_train, epochs=100, batch_size=10)


# In[35]:


y_pred = neural.predict(X_test[:1]) 
neural.evaluate(X_test, y_test)[1]


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


# In[37]:


plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()


# In[38]:


y_pred_NN = (neural.predict(X_test) > 0.5).astype('int')
NN_matrix = confusion_matrix(y_test,y_pred_NN)
NN_matrix


# In[39]:


sns.heatmap(pd.DataFrame(NN_matrix), annot = True, cmap = 'OrRd',fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('confusion matrix for neural network')
plt.ylabel('actual value 0/1',fontsize=12)
plt.xlabel('predict value 0/1',fontsize=12)
plt.show() 


# In[40]:


Classification_accuracy_NN = accuracy_score(y_test,y_pred_NN)
print("Classification accuracy for neural network model is ",Classification_accuracy_NN,'and error rate is',1-Classification_accuracy_NN)


# In[41]:


neural.save('NN.h5')
# from keras.models import load_model
# NN = load_model('NN.h5')
# prediction = NN.predict_classes(Xnew)
# prediction[0]


# # Model 5 : Bagging model

# In[42]:


from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
seed = 7
kfold = model_selection.KFold(n_splits=10,shuffle=True, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
BM = BaggingClassifier(base_estimator=cart,n_estimators=num_trees)
BM.fit(X_train,y_train)
y_pred_BM=BM.predict(X_test)
BM_matrix = confusion_matrix(y_test,y_pred_BM)
BM_matrix


# In[43]:


sns.heatmap(pd.DataFrame(BM_matrix), annot = True, cmap = 'OrRd',fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('confusion matrix for bagging model')
plt.ylabel('actual value 0/1',fontsize=12)
plt.xlabel('predict value 0/1',fontsize=12)
plt.show() 


# In[44]:


Classification_accuracy_BM = accuracy_score(y_test,y_pred_BM)
print("Classification accuracy for bagging model is ",Classification_accuracy_BM,'and error rate is',1-Classification_accuracy_BM)


# In[45]:


pickle.dump(BM, open('BM.pkl','wb'))


# # Model 6 : Random Forest

# In[46]:


from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=100)
RF.fit(X_train,y_train)
y_pred_RF=RF.predict(X_test)
RF_matrix = confusion_matrix(y_test,y_pred_RF)
RF_matrix


# In[47]:


sns.heatmap(pd.DataFrame(RF_matrix), annot = True, cmap = 'OrRd',fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('confusion matrix for bagging model')
plt.ylabel('actual value 0/1',fontsize=12)
plt.xlabel('predict value 0/1',fontsize=12)
plt.show()


# In[48]:


Classification_accuracy_RF = accuracy_score(y_test,y_pred_RF)
print("Classification accuracy for random forest model is ",Classification_accuracy_RF,'and error rate is',1-Classification_accuracy_RF)


# In[49]:


pickle.dump(RF, open('RF.pkl','wb'))

