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
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Fitting
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)