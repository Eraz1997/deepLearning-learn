# import libraries
from sklearn.metrics import confusion_matrix, accuracy_score
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
X_test = sc.fit_transform(X_test)

# now the dataset is ready, and we build the ANN
# init ANN
ann = tf.keras.models.Sequential()
# input layer comes by itself
# we add the first hidden layer.
# this is a dense layer because we have a dense network. in dense networks it is suggested to use "rectifier" activation function for
# the hidden layers.
# how many neuron? there is no rule: you have to experiment and try many times and see how the accuracy improves (this is already done
# by the teacher)
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# this is deep learning, not shallow learning, so we "have to" add another layer (identical to the previous one)
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# now we add the output layer
# there is only one neuron in this layer because our output is a binary variable "client_exits"
# we use sigmoid because it tells us not only if a client is going to exit but with which probability (COOL)
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# remember that "Dense" class creates a layer with fully connected nuerons! We want fully connected neurons for this example
# (dense network)
# if we had more than two categories as possible output values, we would use "softmax" activation function for the output layer

# let's compile the ann: this means building it by specifing the optimizer and other things
# recommended optimizer for this kind of problems is the "adam". the optimizer is the algorithm that applies the gradient descent
# method to improve accuracy from an epoch to the next one
# loss is the way that we compute the loss, of course. if we had more categories we should use "categorical_crossentropy"
# metrics tells wwhat are the metrics that you want to measure and show during the training. In this case we only care about
# the accuracy
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# now we have to train the ann through our train set
# batch_size is actually what you expect it was. now I can understan a little bit more about the training of the rover...
# epochs means max_epochs. An ann has to be trained for a lot of epochs to get good enough. For such a simple network and a simple example we
# train 100 epochs usually improving the ann at each. I can understand that there are a lot of troubles with rover.. maybe is it correlated
# to batch size?
# default batch size is 32 (if you don't want to tune and understand how to do it, let it be 32)
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# let's predict a single customer with the following properties
customer = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])

# let's apply the same transformations to this new row
customer[:, 2] = le.fit_transform(customer[:, 2])
# we call transorm instead of fit_transform because we use mean and variance found at preprocessing step (fit_transform on X_train
# and X_test)
customer = np.array(ct.transform(customer))
customer = sc.transform(customer)
prediction = ann.predict(customer)
print(prediction > 0.5)   # shall we say goodbye to this client?

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# print score of the ann by testing the test set
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
