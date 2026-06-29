from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import SVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#load dataset

data = load_breast_cancer()

X = data.data
y = data.target

print(X.shape)
print(y.shape) #569 samples and 30 features

#label conversion

y = np.where(y == 0, -1, 1) # Convert labels from {0,1} to {-1,+1} for hinge loss

#splitting of data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=7
)

#scaling the datset

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Mean:", np.round(X_train.mean(axis=0)[:5], 2))
print("Std :", np.round(X_train.std(axis=0)[:5], 2))

print(np.unique(y_train))

#train the model

model = SVM(
    learning_rate=0.001,
    lambda_param=0.01,
    n_iters=1000
)
model.fit(X_train, y_train)

#predictions

predictions = model.predict(X_test)

#accuracy

accuracy = np.mean(predictions == y_test)

print(f"Accuracy: {accuracy*100:.2f}%")
