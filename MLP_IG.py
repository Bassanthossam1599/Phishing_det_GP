from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import operator
import timeit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


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

stop = timeit.default_timer()

print("Time = ", stop - start)

clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(train[features], train['Result'])
y_pred=clf.predict(test[features])

print(clf.score(test[features], test['Result']))

acc = accuracy(test['Result'], y_pred)
print("accuracy: ", acc)
fig=plot_confusion_matrix(clf, test[features], test['Result'],display_labels=["Legitimate","phishing"])
fig.figure_.suptitle("Confusion Matrix for our Dataset")
plt.show()