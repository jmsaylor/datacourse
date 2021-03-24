import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('aug_train.csv')

#preprocessing
#imbalanced data - will use under/over-sampling 
#null values
plt.figure(figsize=(5,4))
#sns.heatmap(data.isna(), cmap='plasma', cbar=True) 

#for feature in data.columns:
#    print(feature, " is missing %", np.round(data[feature].isnull().mean() * 100, 3))
    
#sns.countplot(x='gender', data=data)
#sns.countplot(x='relevent_experience', data=data)
sns.countplot(x='enrolled_university', data=data)

#for feature in data.columns:
#    print(feature, ": ", len(data[feature].unique()), " unique")

#got rid of data that seemed too imbalanced 
X = data.iloc[:, [4, 8, 11, 12]] 
y = data.iloc[:, -1]

#relevent experience
exp = pd.get_dummies(X['relevent_experience'], prefix='exp_', drop_first=True)
X = X.drop('relevent_experience', axis=1)
X = pd.concat([exp, X], axis=1)

#experience
rep = {'>20' : '20', '<1':'1'}
X['experience'].replace(rep, inplace=True)
X['experience'] = X['experience'].fillna(value='10')
X['experience'] = X['experience'].astype(float)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['experience'] = le.fit_transform(X['experience'])

#last_new_job
rep = {'>4':5, 'never': 0, 'nan': 3 }
X['last_new_job'].replace(rep, inplace=True)
X['last_new_job'] = X['last_new_job'].fillna(value='3')
X['last_new_job'] = X['last_new_job'].astype(float)
unique = X['last_new_job'].unique()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)
