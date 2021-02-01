from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

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

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(train[features], y)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
              solver='lbfgs')
preds=clf.predict(test[features])
acc= accuracy(test['Result'],preds)
print("accuracy: ",acc)