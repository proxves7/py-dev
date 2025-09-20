import pandas as pd
pd.options.mode.chained_assignment = None

titanic = pd.read_csv("datasets/train.csv")
print(titanic)

X = titanic[['Pclass', 'Age', 'Sex']]
y = titanic['Survived']

X['Age'] = X['Age'].fillna(X['Age'].mean())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
print(X_train)

X_train['Sex'] = X_train['Sex'].map({'male':0, 'female':1})
X_test['Sex'] = X_test['Sex'].map({'male':0, 'female':1})
print(X_train)

#####
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print('The accuracy of Random Forest Classifier on testing set:', rfc.score(X_test, y_test))

#####
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
print('The accuracy of eXtreme Gradient Boosting Classifier on testing set:', xgbc.score(X_test, y_test))
