import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_predictions = regressor.predict(X_test)

from sklearn.metrics import r2_score

sample_size, indep_vars = X_train.shape

r2 = sklearn.metrics.r2_score(y_test, y_predictions)

#Not needed for single linear regression
#r2_adjusted = 1-(1-r2)*(sample_size - 1)/(sample_size - indep_vars - 1)


plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

#plt_test.scatter(X_test, y_test, color = 'purple')
#plt_test.plot(X_train, regressor.predict(X_train), color = 'yellow')
#plt_test.title('Salary vs. Experience (Test Set)')
#plt_test.xlabel('Years of Experience')
#plt_test.ylabel('Salary')