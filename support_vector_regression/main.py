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

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(np.array(y).reshape(-1,1))

from sklearn.svm import SVR
svr_regressor = SVR(kernel ='rbf')
svr_regressor.fit(X, y)

prediction = svr_regressor.predict(sc_X.transform(np.array(6.5).reshape(-1,1)))
prediction = sc_y.inverse_transform(prediction)


plt.title('SVR')
plt.scatter(X, y, color='purple')
plt.plot(X, svr_regressor.predict(X), color = 'blue')
plt.ylabel('Salary')
plt.xlabel('Years')