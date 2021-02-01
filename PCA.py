# Import dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Initialize process time list for code timing across entire program

# Initialize process time list for code timing across entire program
#process_time = []

# Import train and test datasets


df = pd.read_csv("dataset.csv")
df.head()
df= df.drop('index', axis=1)


#X.head()
# Splitting the dataset into the Training set and Test set


X=df.drop('Result',1)

Y=df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Normalizing feature set
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Applying PCA
pca=PCA()
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)

explained_variance= pca.explained_variance_ratio_

#The transformed data has been reduced to a single dimension. To understand the effect of this dimensionality reduction
pca = PCA(n_components=20)
#pca.fit(X_train)
#X_pca = pca.transform(X_train)
#print("original shape:   ", X.shape)
#print("transformed shape:", X_pca.shape)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


classifier = RandomForestClassifier(max_depth=15, random_state=1, bootstrap=False, max_features="auto", n_estimators=50)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)



cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))