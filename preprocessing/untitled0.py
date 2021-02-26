import numpy as np
import matplotlib.pyplot as plt
import sklearn.impute
import sklearn.preprocessing
import sklearn.compose
import pandas as pd
    

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3:].values

imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean', )
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])



country_transformer = sklearn.compose.ColumnTransformer(
    transformers= [
        ("Country",
            sklearn.preprocessing.OneHotEncoder(),
            [0])
        ], remainder='passthrough'
    )
X = country_transformer.fit_transform(X)

label_encoder = sklearn.preprocessing.LabelEncoder()
purchased_encoder = label_encoder.fit(y)
y = purchased_encoder.transform(y)


# X[:,0] = label_encoder.fit_transform(X[:, 0])

# one_hot_encoder = sklearn.preprocessing.OneHotEncoder().fit(X)
# X[:,0] = one_hot_encoder.fit_transform(X[:,0]).toarray()