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
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .7

train, test = df[df['is_train'] == True], df[df['is_train'] == False]
features = df.columns[0:30]
y = train['Result']


clf = RandomForestClassifier(random_state=1, bootstrap=False, max_features="auto", n_estimators=50)


clf.fit(train[features], y)
# apply the trained classifier to the test
preds = clf.predict(test[features])
acc = accuracy(test['Result'], preds)
print("accuracy: ", acc)

