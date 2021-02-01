from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, validation_curve
import matplotlib.pyplot as plt

np.random.seed(0)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


df = pd.read_csv("dataset.csv")
df = df.drop('index', axis=1)
features = df.iloc[:, 0:30].values
x_train, x_test, y_train, y_test = train_test_split(features, df.iloc[:, 30:31].values, train_size=0.65)
y_train = y_train.ravel()
y_test = y_test.ravel()

clf = RandomForestClassifier(random_state=1, bootstrap=False, max_features="auto", n_estimators=50)

clf.fit(x_train, y_train)  #

preds = clf.predict(x_test)
acc = accuracy(y_test, preds)
print("accuracy: ", acc)
