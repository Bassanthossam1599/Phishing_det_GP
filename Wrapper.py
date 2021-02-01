# import Sep as Sep
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# name = ['having_IPhaving_IP_Address','URLURL_Length','Shortining_Service','having_At_Symbol',
# 'double_slash_redirecting','Prefix_Suffix', 'having_Sub_Domain','SSLfinal_State','Domain_registeration_length',
# 'Favicon','port','HTTPS_token',	'Request_URL','URL_of_Anchor', 'Links_in_tags','SFH','Submitting_to_email',
# 'Abnormal_URL','Redirect','on_mouseover',	'RightClick','popUpWidnow','Iframe', 'age_of_domain','DNSRecord'	,
# 'web_traffic',	'Page_Rank','Google_Index',	'Links_pointing_to_page','Statistical_report'] Read data
df = pd.read_csv("dataset.csv")
df = df.drop('index', axis=1)
features = df.iloc[:, 0:30].values

x_train, x_test, y_train, y_test = train_test_split(features, df.iloc[:, 30:31].values, train_size=0.7, shuffle=False)
# x = df.values[:, :-1]
# y = df.values[:, -1:]

# df.dropna(axis=0, how="any", inplace=True)
# print(x)
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
y_train = y_train.ravel()
y_test = y_test.ravel()
# Build RF classifier to use in feature selection
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
# Build step forward feature selection
sfs1 = sfs(clf,
           k_features=18,
           forward=True,
           # The floating algorithms have an additional exclusion or inclusion step to remove features once they were
           # included (or excluded), so that a larger number of feature subset combinations can be sampled
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)

# Perform SFFS
sfs1 = sfs1.fit(x_train, y_train)
# Which features?
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)