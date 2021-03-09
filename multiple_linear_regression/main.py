import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

state_transformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')

X = state_transformer.fit_transform(X);

#avoiding dummy variable trap, even though many frameworks automatically do this
X = X[:, 1:].astype('float64')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Fitting
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.summary())

predictions = regressor.predict(X_test)

#Backward Elimination
regressor_backward = LinearRegression()
X_backward = X_train[:, [2,4]]
regressor_backward.fit(X_backward, y_train)
X_test_backward = X_test[:, [2, 4]]
predictions_backward = regressor_backward.predict(X_test_backward)

from statsmodels.regression.linear_model import OLS

import statsmodels.formula.api as smf
import statsmodels.api as sm
#but...we add it back, in a way, for analysis. b * 1 = b
X = np.append(arr = np.ones((len(X), 1)).astype('float64'), values = X, axis=1)
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = OLS(endog = y, exog = X_optimal).fit()

X_optimal = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = OLS(endog = y, exog = X_optimal).fit()

X_optimal = X[:, [0, 3, 4, 5]]
regressor_OLS = OLS(endog = y, exog = X_optimal).fit()

X_optimal = X[:, [0, 3, 5]]
regressor_OLS = OLS(endog = y, exog = X_optimal).fit()

print(regressor_OLS.summary())