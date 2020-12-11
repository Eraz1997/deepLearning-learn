# import libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


# read csv
dataset = pd.read_csv('part1/Churn_Modelling.csv')
# delete y and rowNumber and customerId and surname
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values  # get only score (y)

# Encode categorical data (Gender and similar strings must be converted into numbers)
# encode gender with labelencoding
# labelencoding replaces "male" with 0/1 and "female" with 1/0
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])  # encode "gender"
# encode geography with hotencoding
# hotencoding replaces geography with N features (N = cardinality of geography)
# so "france" is 1 0 0, "spain" is 0 1 0, "germany" is 0 0 1
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# split into train set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# feature scaling: standardize all features
# without standard scaling, a feature would be more important than another at start
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_train = sc.fit_transform(X_test)

# now the dataset is ready, and we build the ANN
