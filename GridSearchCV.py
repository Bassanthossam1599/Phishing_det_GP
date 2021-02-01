from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import operator
import timeit
from sklearn.model_selection import GridSearchCV

np.random.seed(0)

start = timeit.default_timer()

df = pd.read_csv("dataset.csv")
df = df.drop('index', axis=1)
name = df.columns[0:30]


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def Entropy(result_col):
    (values, counts) = np.unique(result_col, return_counts=True)
    entropy = np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(values))])
    return entropy


def InformationGain(data, feature_name, result_name="Result"):
    total_entropy = Entropy(data[result_name])
    (vals, counts) = np.unique(data[feature_name], return_counts=True)
    condintional_entropy = np.sum(
        [(counts[i] / np.sum(counts)) * Entropy(data.where(data[feature_name] == vals[i]).dropna()[result_name])
         for i in range(len(vals))])
    gain = total_entropy - condintional_entropy
    return gain


def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)

    return list


gaindic = {}
for i in range(30):
    gain = InformationGain(df, name[i], "Result")
    gaindic[i] = gain
    # print(gain)

sortedtubles = sorted(gaindic.items(), key=operator.itemgetter(1), reverse=True)
# print(sortedtubles)
sorteddic = {k: v for k, v in sortedtubles}
# print(sorteddic)
# print(sorteddic.keys())
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# creating data frames to the test and train rows
train, test = df[df['is_train'] == True], df[df['is_train'] == False]

keyList = getList(sorteddic)
# print(keyList)
# creating a list of feature names
features = df.columns[keyList]

# getting the target column into y
y = train['Result']
'''
# num of trees in random forest
n_estimators = [10, 50]
# number of features to consider at every split
max_features = ['auto', 'sqrt']
# max number of level of levels in tree
max_depth = [2, 4, 20]
# min number of samples requires to split a node
min_samples_split = [2, 5]
# min number of samples required at each leaf node
min_sample_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]
param_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'random_state': [0, 1],

              'bootstrap': bootstrap
              }
'''
n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

param_grid = dict(n_estimators=n_estimators, max_depth=max_depth,
                  min_samples_split=min_samples_split,
                  min_samples_leaf=min_samples_leaf)

# creating RF classifier

clf = RandomForestClassifier()
cv = GridSearchCV(clf, param_grid, cv=3, verbose=1,
                  n_jobs=-1)

# training the classifier
cv.fit(train[features], y)
print(cv.best_params_)
