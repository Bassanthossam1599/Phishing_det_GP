from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

np.random.seed(0)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def WrapperAlgo(x_train, y_train):
    clsf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    # Build step forward feature selection
    sfs1 = sfs(clsf,
               k_features=18,
               forward=True,
               # The floating algorithms have an additional exclusion or inclusion step to remove features once they
               # were included (or excluded), so that a larger number of feature subset combinations can be sampled
               floating=False,
               verbose=2,
               scoring='accuracy',
               cv=5)

    # Perform SFFS
    sfs1 = sfs1.fit(x_train, y_train)
    # Which features?
    feat_cols = list(sfs1.k_feature_idx_)
    return feat_cols


df = pd.read_csv("dataset.csv")
df = df.drop('index', axis=1)
features = df.iloc[:, 0:30].values
x_train, x_test, y_train, y_test = train_test_split(features, df.iloc[:, 30:31].values, train_size=0.7)  # outputs a numpy arrays

# converting numpy arrays into df to access columns with their names
features_names = df.columns[:30]
x_train_df = pd.DataFrame(x_train, columns=features_names)
x_test_df = pd.DataFrame(x_test, columns=features_names)

# converting result column into 1d array to send in prediction "right format"
y_train = y_train.ravel()
y_test = y_test.ravel()

# performing wrapper
MyList = WrapperAlgo(x_train, y_train)

selected_features = df.columns[MyList]  # names only

# getting new dataframes with the selected columns by wrapper
new_xtrain = x_train_df[selected_features]
new_xtest = x_test_df[selected_features]

clf = RandomForestClassifier(random_state=1, bootstrap=False, max_features="auto", n_estimators=50)
clf.fit(new_xtrain, y_train)  # adding a print will show every attribute in the algorithm
# apply the trained classifier to the test
preds = clf.predict(new_xtest)


acc = accuracy(y_test, preds)
print("accuracy: ", acc)
