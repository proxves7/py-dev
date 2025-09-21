import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
from interpret import show
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from interpret.data import Marginal, ClassHistogram
from e0_data import Cervical_DataLoader, Bike_DataLoader
import time

# from interpret.provider import InlineProvider
# from interpret import set_visualize_provider
# set_visualize_provider(InlineProvider)


### Classification Problem
## Explore the dataset - Risk Factors for Cervical Cancer
data_loader = Cervical_DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
print(data_loader.data)

X_train, X_test, y_train, y_test = data_loader.get_data_split()
print(X_train.shape)
print(X_test.shape)

print("Before oversampling, X_train: ", X_train.shape, " y_train:", y_train.shape)
hist = ClassHistogram().explain_data(X_train, y_train, name='Train Data')
show(hist)

X_train, y_train = data_loader.oversample(X_train, y_train)
print("After oversampling, X_train: ", X_train.shape, " y_train:", y_train.shape)
hist = ClassHistogram().explain_data(X_train, y_train, name = 'Train Data')
show(hist)


## Classification Moldes
from interpret.glassbox import (LogisticRegression,
                                ClassificationTree,
                                ExplainableBoostingClassifier)


## Logistic Regression Model
lr = LogisticRegression(random_state=2022, feature_names=X_train.columns, penalty='l1', solver='liblinear')
lr.fit(X_train, y_train)
print("Training finished.")

y_pred = lr.predict(X_test)
print(f"F1 Score: {round(f1_score(y_test, y_pred, average='macro'), 2)}")
print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 2)}")

print("Explain global logistic regression model")
lr_global = lr.explain_global(name='Logistic Regression')
show(lr_global)

print("Explain local logistic regression model")
lr_local = lr.explain_local(X_test, y_test, name='Logistic Regression')
show(lr_local)


## Decision Tree for Classification
ct = ClassificationTree(random_state=2022)
ct.fit(X_train, y_train)
print("Training finished.")

y_pred = ct.predict(X_test)
print(f"F1 Score: {round(f1_score(y_test, y_pred, average='macro'), 2)}")
print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 2)}")

print("Explain global decision tree for classification model")
ct_global = ct.explain_global(name='Classification Tree')
show(ct_global)

print("Explain local decision tree for classification model")
ct_local = ct.explain_local(X_test, y_test, name='Classification Tree')
show(ct_local)


## Explainable Boosting Machine for Classification
ebc = ExplainableBoostingClassifier(random_state=2022)
ebc.fit(X_train, y_train) 
print("Training finished.")

y_pred = ebc.predict(X_test)
print(f"F1 Score: {round(f1_score(y_test, y_pred, average='macro'), 2)}")
print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 2)}")

print("Explain global explainable boosting machine for classification")
ebc_global = ebc.explain_global(name='Classification EBM')
show(ebc_global)

print("Explain local explainable boosting machine for classification")
ebc_local = ebc.explain_local(X_test, y_test, name='Classification EBM')
show(ebc_local)

##==================================================
### Regression Problem
data_loader = Bike_DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
data_loader.data

X_train, X_test, y_train, y_test = data_loader.get_data_split()
print(X_train.shape)
print(X_test.shape)

print("X_train: ", X_train.shape, " y_train:", y_train.shape)
marginal = Marginal().explain_data(X_train, y_train, name = 'Train Data')
show(marginal)


## Regression Moldes
from interpret.glassbox import (LinearRegression,
                                RegressionTree, 
                                ExplainableBoostingRegressor)


## Linear Regression Model
lr = LinearRegression(random_state=2022, feature_names=X_train.columns)
lr.fit(X_train, y_train)
print("Training finished.")

y_pred = lr.predict(X_test)
print(f"Root Mean Squared Error: {round(mean_squared_error(y_test, y_pred)**(1/2), 2)}")
print(f"R2: {round(r2_score(y_test, y_pred), 2)}")

print("Explain global linear regression model")
lr_global = lr.explain_global(name='Linear Regression')
show(lr_global)

print("Explain local linear regression model")
lr_local = lr.explain_local(X_test, y_test, name='Linear Regression')
show(lr_local)


## Decision Tree for Regression
rt = RegressionTree(random_state=2022)
rt.fit(X_train, y_train)
print("Training finished.")

y_pred = rt.predict(X_test)
print(f"Root Mean Squared Error: {round(mean_squared_error(y_test, y_pred)**(1/2), 2)}")
print(f"R2: {round(r2_score(y_test, y_pred), 2)}")

print("Explain global decision tree model for regression")
rt_global = rt.explain_global(name='Regression Tree')
show(rt_global)

print("Explain local decision tree model for regression")
rt_local = rt.explain_local(X_test, y_test, name='Regression Tree')
show(rt_local)


## Explainable Boosting Machine for Regression
ebr = ExplainableBoostingRegressor(random_state=2022)
ebr.fit(X_train, y_train) 
print("Training finished.")

y_pred = ebr.predict(X_test)
print(f"Root Mean Squared Error: {round(mean_squared_error(y_test, y_pred)**(1/2), 2)}")
print(f"R2: {round(r2_score(y_test, y_pred), 2)}")

print("Explain global explainable boosting machine for regression")
ebr_global = ebr.explain_global(name='Regression EBM')
show(ebr_global)

print("Explain local explainable boosting machine for regression")
ebr_local = ebr.explain_local(X_test, y_test, name='Regression EBM')
show(ebr_local)

##==================================================
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Dashboard shutdown")
