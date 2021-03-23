import numpy as np
import pandas as pd
import sklearn

data = pd.read_csv('aug_train.csv', nrows=1000)

#got rid of data that seemed too imbalanced 
X = data.iloc[:, [3,4,5,6,8,11,12]] 
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

#preprocessing
#imbalanced data - will use under/over-sampling 
#null values 