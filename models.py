import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras import metrics

import pickle

from matplotlib import pyplot as plt

from pathlib import Path

""" 
Please run the data_cleaning script ONCE before running this one

X thanks X
"""

Path("models").mkdir(parents=True, exist_ok=True)

df = pd.read_csv('LOL/Clean_LeagueofLegends.csv')

x_cols = ['blueMiddleChamp', 'blueJungleChamp', 'redMiddleChamp', 'redJungleChamp',
          'rKills_pre15', 'rTowers_pre15', 'rDragons_pre15','rHeralds_pre15', 'golddiff_min15']

x_cols_v2 = ['blueJungleChampTags', 'redJungleChampTags', 'blueMiddleChampTags', 'redMiddleChampTags', 
            'blueADCChampTags', 'redADCChampTags', 'blueSupportChampTags', 'redSupportChampTags', 
            'blueTopChampTags', 'redTopChampTags', 'rKills_pre15', 'rTowers_pre15', 'rDragons_pre15',
            'rHeralds_pre15', 'golddiff_min15']

x_cols_v3 = ['blueJungleChampTags', 'redJungleChampTags', 'blueMiddleChampTags', 'redMiddleChampTags', 
            'blueADCChampTags', 'redADCChampTags', 'blueSupportChampTags', 'redSupportChampTags', 
            'blueTopChampTags', 'redTopChampTags', 'rKills_pre15', 'rTowers_pre15', 'rDragons_pre15',
            'rHeralds_pre15']

x = df[x_cols_v2]
y = df['rResult']

x = pd.get_dummies(x, columns=['blueJungleChampTags', 'redJungleChampTags', 'blueMiddleChampTags', 'redMiddleChampTags', 
                                'blueADCChampTags', 'redADCChampTags', 'blueSupportChampTags', 'redSupportChampTags', 
                                'blueTopChampTags', 'redTopChampTags'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

'''
Logistic model

Variables:
    redJungleChampTags: Red Champion selection for the Jungle role (string) 
    redMiddlehampTags: Red Champion selection for the Middle role (string) 
    redTopChampTags: Red Champion selection for the Top role (string) 
    redADCChampTags: Red Champion selection for the ADC role (string) 
    redSupportChampTags: Red Champion selection for the Support role (string) 
    blueJungleChampTags: Blue Champion selection for the Jungle role (string)
    blueMiddlehampTags: Blue Champion selection for the Middle role (string)
    blueTopChampTags: Blue Champion selection for the Top role (string)
    blueADCChampTags: Blue Champion selection for the ADC role (string)
    blueSupportChampTags: Blue Champion selection for the Support role (string)
    Red_Kills_pre15: number of kills before minute 15 for red (integer)
    Red_Towers_pre15: number of towers destroyed before minute 15 for red (integer)
    Red_dragons_pre15: number of dragons captured by minute 15 (integer)
    Red_herald_pre15: herald captured by minute 15 for red (integer)
    Gold_Diff_MIN_15: Diff in gold (red team - blue team) (integer)

Target:
    rResult: 1 if red team won (boolean)
'''

logit = LogisticRegression()
logit.fit(x_train, y_train)

y_pred = logit.predict(x_test)

logit_accuracy = sum(y_pred==y_test)/len(y_test)

print('Logistic model accuracy: {}'.format(logit_accuracy)) # 0.7355643044619422
print('='*50 + 'Confusion matrix' + '='*50, '\n', confusion_matrix(y_test, y_pred))

pickle.dump(logit, open('models/logit.pkl', 'wb'))

titles_options = [("Logit Confusion matrix, without normalization", None),
                  ("Logit Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(logit, x_test, y_test,
                                 display_labels=['Victory', 'Defeat'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

# plt.show()

'''
Decision Tree

Variables:
    redJungleChampTags: Red Champion selection for the Jungle role (string) 
    redMiddlehampTags: Red Champion selection for the Middle role (string) 
    redTopChampTags: Red Champion selection for the Top role (string) 
    redADCChampTags: Red Champion selection for the ADC role (string) 
    redSupportChampTags: Red Champion selection for the Support role (string) 
    blueJungleChampTags: Blue Champion selection for the Jungle role (string)
    blueMiddlehampTags: Blue Champion selection for the Middle role (string)
    blueTopChampTags: Blue Champion selection for the Top role (string)
    blueADCChampTags: Blue Champion selection for the ADC role (string)
    blueSupportChampTags: Blue Champion selection for the Support role (string)
    Red_Kills_pre15: number of kills before minute 15 for red (integer)
    Red_Towers_pre15: number of towers destroyed before minute 15 for red (integer)
    Red_dragons_pre15: number of dragons captured by minute 15 (integer)
    Red_herald_pre15: herald captured by minute 15 for red (integer)
    Gold_Diff_MIN_15: Diff in gold (red team - blue team) (integer)

Target:    
    rResult: 1 if red team won (boolean)
'''

dTree = DecisionTreeClassifier()

dTree.fit(x_train, y_train)

y_pred = dTree.predict(x_test)

dTree_accuracy = sum(y_pred==y_test)/len(y_test)

print('Decision tree model accuracy: {}'.format(dTree_accuracy)) # 0.6437007874015748
print('='*50 + 'Confusion matrix' + '='*50, '\n', confusion_matrix(y_test, y_pred))

pickle.dump(dTree, open('models/dTree.pkl', 'wb'))

titles_options = [("Decision tree Confusion matrix, without normalization", None),
                  ("Decision tree Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(dTree, x_test, y_test,
                                 display_labels=['Victory', 'Defeat'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

# plt.show()
'''
Random Forest

Variables:
    redJungleChampTags: Red Champion selection for the Jungle role (string) 
    redMiddlehampTags: Red Champion selection for the Middle role (string) 
    redTopChampTags: Red Champion selection for the Top role (string) 
    redADCChampTags: Red Champion selection for the ADC role (string) 
    redSupportChampTags: Red Champion selection for the Support role (string) 
    blueJungleChampTags: Blue Champion selection for the Jungle role (string)
    blueMiddlehampTags: Blue Champion selection for the Middle role (string)
    blueTopChampTags: Blue Champion selection for the Top role (string)
    blueADCChampTags: Blue Champion selection for the ADC role (string)
    blueSupportChampTags: Blue Champion selection for the Support role (string)
    Red_Kills_pre15: number of kills before minute 15 for red (integer)
    Red_Towers_pre15: number of towers destroyed before minute 15 for red (integer)
    Red_dragons_pre15: number of dragons captured by minute 15 (integer)
    Red_herald_pre15: herald captured by minute 15 for red (integer)
    Gold_Diff_MIN_15: Diff in gold (red team - blue team) (integer)

Target:    
    rResult: 1 if red team won (boolean)

Number of trees:
    100
'''

rForest = RandomForestClassifier(n_estimators=100)

# x_train = StandardScaler().fit_transform(x_train)
# x_test = StandardScaler().fit_transform(x_test)
# print(x_train)

rForest.fit(x_train, y_train)

y_pred = rForest.predict(x_test)
y_pred_P = rForest.predict_proba(x_test)

rForest_accuracy = sum(y_pred==y_test)/len(y_test)

print('Random Forest model accuracy: {}'.format(rForest_accuracy)) # 0.7276902887139107
print('='*50 + 'Confusion matrix' + '='*50, '\n', confusion_matrix(y_test, y_pred))

pickle.dump(rForest, open('models/rForestM2.pkl', 'wb'))

titles_options = [("Random forest Confusion matrix, without normalization", None),
                  ("Random forest Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(rForest, x_test, y_test,
                                 display_labels=['Victory', 'Defeat'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

# plt.show()
'''
SVM

Variables:
    redJungleChampTags: Red Champion selection for the Jungle role (string) 
    redMiddlehampTags: Red Champion selection for the Middle role (string) 
    redTopChampTags: Red Champion selection for the Top role (string) 
    redADCChampTags: Red Champion selection for the ADC role (string) 
    redSupportChampTags: Red Champion selection for the Support role (string) 
    blueJungleChampTags: Blue Champion selection for the Jungle role (string)
    blueMiddlehampTags: Blue Champion selection for the Middle role (string)
    blueTopChampTags: Blue Champion selection for the Top role (string)
    blueADCChampTags: Blue Champion selection for the ADC role (string)
    blueSupportChampTags: Blue Champion selection for the Support role (string)
    Red_Kills_pre15: number of kills before minute 15 for red (integer)
    Red_Towers_pre15: number of towers destroyed before minute 15 for red (integer)
    Red_dragons_pre15: number of dragons captured by minute 15 (integer)
    Red_herald_pre15: herald captured by minute 15 for red (integer)
    Gold_Diff_MIN_15: Diff in gold (red team - blue team) (integer)

Target:    
    rResult: 1 if red team won (boolean)
'''

svm = SVC(probability=True)

svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)
y_pred_p = svm.predict_proba(x_test)

print(x_test.head(1))

print(svm.predict_proba(x_test.head(1))[0])

svm_accuracy = sum(y_pred==y_test)/len(y_test)

print('SVM model accuracy: {}'.format(svm_accuracy)) # 0.7368766404199475
print('='*50 + 'Confusion matrix' + '='*50, '\n', confusion_matrix(y_test, y_pred))

pickle.dump(svm, open('models/svm.pkl', 'wb'))

titles_options = [("SVM Confusion matrix, without normalization", None),
                  ("SVM Normalized confusion matrix", 'true')]

# for title, normalize in titles_options:
#     disp = plot_confusion_matrix(svm, x_test, y_test,
#                                  display_labels=['Victory', 'Defeat'],
#                                  cmap=plt.cm.Blues,
#                                  normalize=normalize)
#     disp.ax_.set_title(title)

# plt.show()

# print(y_pred_p[:,1])

# plt.scatter(x_test['golddiff_min15'], y_pred_p[:,0])
# plt.title('Gold difference at minute 15 vs winning probability')
# plt.xlabel('Gold difference')
# plt.ylabel('Winning probability')
# plt.show()
'''
Neural Network

Variables:
    redJungleChampTags: Red Champion selection for the Jungle role (string) 
    redMiddlehampTags: Red Champion selection for the Middle role (string) 
    redTopChampTags: Red Champion selection for the Top role (string) 
    redADCChampTags: Red Champion selection for the ADC role (string) 
    redSupportChampTags: Red Champion selection for the Support role (string) 
    blueJungleChampTags: Blue Champion selection for the Jungle role (string)
    blueMiddlehampTags: Blue Champion selection for the Middle role (string)
    blueTopChampTags: Blue Champion selection for the Top role (string)
    blueADCChampTags: Blue Champion selection for the ADC role (string)
    blueSupportChampTags: Blue Champion selection for the Support role (string)
    Red_Kills_pre15: number of kills before minute 15 for red (integer)
    Red_Towers_pre15: number of towers destroyed before minute 15 for red (integer)
    Red_dragons_pre15: number of dragons captured by minute 15 (integer)
    Red_herald_pre15: herald captured by minute 15 for red (integer)
    Gold_Diff_MIN_15: Diff in gold (red team - blue team) (integer)

Target:
    rResult: 1 if red team won (boolean)

Architecture:
    * dense layer: 10 relu
    * dense layer: 10 relu
    * dense layer: 10 relu
    * dense layer: 5 relu
    * dense layer: 1 Sigmoid
'''

scalar = MinMaxScaler()
scalar.fit(x)

x_nn = scalar.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_nn, y, test_size=0.2, random_state=0)

nn = Sequential()
print('xtrain lenght: {}'.format(len(x_train[0])))
nn.add(Dense(10, input_dim=len(x_train[0]), activation='relu'))
nn.add(Dense(10, activation='relu'))
nn.add(Dense(10, activation='relu'))
nn.add(Dense(5, activation='relu'))
nn.add(Dense(1, activation='sigmoid'))

nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn.summary()

history = nn.fit(x_train, y_train, epochs=150, batch_size=10)

print(history.history.keys())

nn.evaluate(x_test, y_test) # 0.6640
y_pred = nn.predict(x_test)

nn.save('models/nn')

'''
Linear regression

Variables:
    redJungleChampTags: Red Champion selection for the Jungle role (string) 
    redMiddlehampTags: Red Champion selection for the Middle role (string) 
    redTopChampTags: Red Champion selection for the Top role (string) 
    redADCChampTags: Red Champion selection for the ADC role (string) 
    redSupportChampTags: Red Champion selection for the Support role (string) 
    blueJungleChampTags: Blue Champion selection for the Jungle role (string)
    blueMiddlehampTags: Blue Champion selection for the Middle role (string)
    blueTopChampTags: Blue Champion selection for the Top role (string)
    blueADCChampTags: Blue Champion selection for the ADC role (string)
    blueSupportChampTags: Blue Champion selection for the Support role (string)
    Red_Kills_pre15: number of kills before minute 15 for red (integer)
    Red_Towers_pre15: number of towers destroyed before minute 15 for red (integer)
    Red_dragons_pre15: number of dragons captured by minute 15 (integer)
    Red_herald_pre15: herald captured by minute 15 for red (integer)
    Gold_Diff_MIN_15: Diff in gold (red team - blue team) (integer)

Target:
    final_golddiff: Gold difference at the end of the game
'''

x = df[x_cols]
x = pd.get_dummies(x,columns=['blueMiddleChamp', 'blueJungleChamp', 'redMiddleChamp', 'redJungleChamp'])

y = df['golddiff_final']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

y_pred = linear_model.predict(x_test)

print('linear model R^2: {}'.format(linear_model.score(x_test, y_test))) # 0.4371161384443095

pickle.dump(linear_model, open('models/linear_model.pkl', 'wb'))

plt.scatter(y_pred, y_test)
plt.plot(    
    np.arange(min(y_test.values), max(y_test.values)), 
    np.arange(min(y_test.values), max(y_test.values)), 
    c='red',
    linewidth=2
    )
plt.title('Predicted gold diff vs Real gold diff')
plt.xlabel('Predicted gold diff')
plt.ylabel('Real gold diff')
# plt.show()