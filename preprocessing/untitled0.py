import numpy as np
import matplotlib.pyplot as plt
import sklearn.impute
import sklearn.preprocessing
import sklearn.compose
import sklearn.model_selection

import pandas as pd
    

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3:].values


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

"""imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean', )
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])"""

"""country_transformer = sklearn.compose.ColumnTransformer(
    transformers= [
        ("Country",
            sklearn.preprocessing.OneHotEncoder(),
            [0])
        ], remainder='passthrough'
    )
X = country_transformer.fit_transform(X)"""

"""label_encoder = sklearn.preprocessing.LabelEncoder()
purchased_encoder = label_encoder.fit(y)
y = purchased_encoder.transform(y)"""


"""sc_X = sklearn.preprocessing.StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) """
