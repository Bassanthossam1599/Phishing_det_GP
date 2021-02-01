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

# clf = RandomForestClassifier(random_state=1)
n_estimators = [10, 100, 300, 500, 750, 800, 1200, 2000]
max_depth = [5, 10, 15, 20, 25, 30]
min_sample_split = [5, 10, 15, 20, 25]
train_scores, test_scores = validation_curve(
    RandomForestClassifier(),
    X=x_train, y=y_train,
    param_name='max_depth',
    param_range=max_depth, cv=3)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with Random forest")
plt.xlabel("values of hyperparameter max_depth")
plt.ylabel("Accuracy Score")

lw = 2
plt.semilogx(max_depth, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(max_depth, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(max_depth, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(max_depth, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


