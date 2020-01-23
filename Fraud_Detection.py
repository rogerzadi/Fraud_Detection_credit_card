import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

df = pd.read_csv('../creditcard.csv')
print(df.info())
print(df.Class.value_counts()) #Lo primero que podemos observar es que son muy pocos datos de fraude positivo (solo el .173%)

y=df.Class
X=df.drop(columns=['Class'])

X_scaled = scale(X)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
X_train, X_test, y_train, y_test=train_test_split(X_reduced, y, test_size=0.2)

rf=RFC()
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)

from sklearn.metrics import accuracy_score as acc

print(acc(y_test, y_pred))

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_test,y_pred))

X_train2, X_test2, y_train2, y_test2=train_test_split(X, y, test_size=0.2)
rf2=RFC()
rf2.fit(X_train2, y_train2)

feats = {}
for feature, importance in zip(X_train2.columns, rf2.feature_importances_):
    feats[feature] = importance

importances = pd.DataFrame.from_dict(feats, orient="index").rename(
    columns={0: "importance"}
)
imp = importances.sort_values(by="importance", ascending=False)[:20]
print(imp)


