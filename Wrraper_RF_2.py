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

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# creating data frames to the test and train rows
train, test = df[df['is_train'] == True], df[df['is_train'] == False]

# performing wrapper
MyList = WrapperAlgo(train[features], train['Result'])

selected_features = df.columns[MyList]  # names only

# getting new dataframes with the selected columns by wrapper

clf = RandomForestClassifier(n_jobs=2, random_state=0)

clf.fit(train[selected_features], train['Result'])  # adding a print will show every attribute in the algorithm
# apply the trained classifier to the test
preds = clf.predict(test[selected_features])


acc = accuracy(test['Result'], preds)
print("accuracy: ", acc)
