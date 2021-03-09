import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #vector v. matrix
#x should be a matrix, and y a vector
y = dataset.iloc[:, 2].values

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

prediction = regressor.predict(np.array(6.5).reshape(-1,1))

import sklearn.metrics as metrics
print(metrics.classification_report(y, regressor.predict(X)))

#Visualizing (for high resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='blue')
plt.plot(X_grid, regressor.predict(X_grid), color='purple')
plt.title('Decision Tree Regression (non-linear)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

