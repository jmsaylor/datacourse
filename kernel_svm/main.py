import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler().fit(X_train)
X_train = scale_x.transform(X_train)
X_test = scale_x.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np. arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))

X1_shape, X2_shape = X1.shape, X2.shape

predictions = classifier.predict(np.array([X1.ravel(), X2.ravel()], dtype=float).T).reshape(X1.shape)

plt.contourf(X1, X2, predictions,
             alpha = 0.75, cmap = ListedColormap(('red','green')))
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()
