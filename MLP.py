
from sklearn.neural_network import MLPClassifier

import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


df = pd.read_csv("dataset.csv")
df = df.drop('index', axis=1)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

train, test = df[df['is_train'] == True], df[df['is_train'] == False]
features = df.columns[0:30]
y = train['Result']
clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(train[features], train['Result'])
y_pred=clf.predict(test[features])
print(clf.score(test[features], test['Result']))
fig=plot_confusion_matrix(clf, test[features], test['Result'],display_labels=["Legitimate","phishing"])
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()