import time
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from model import KNN

"""from plots import (
    plot_accuracy,
    plot_prediction_time,
    plot_confusion_matrix
)"""
#Load Dataset

data = load_breast_cancer()

X = data.data
y = data.target

#train test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=7
)

#feature scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#knn from scratch

print("KNN from scratch\n")

scratch_model = KNN(
    k=5,
    distance="euclidean"
)

start = time.perf_counter()

scratch_model.fit(X_train, y_train)

end = time.perf_counter()

scratch_train_time = end - start

#prediction

iterations = 100
start = time.perf_counter()
for _ in range(iterations):
    scratch_predictions = scratch_model.predict(X_test)

end = time.perf_counter()

scratch_prediction_time = (end - start) / iterations

#accuracy

scratch_accuracy = np.mean(
    scratch_predictions == y_test
)

print(f"Accuracy : {scratch_accuracy*100:.2f}%")

print(f"Prediction Time : {scratch_prediction_time:.6f} sec")

#confusion matrix

cm_ours = confusion_matrix(
    y_test,
    scratch_predictions
)

#sklearn knn

print("\nScikit-Learn KNN\n")

sk_model = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
)

#fit

start = time.perf_counter()

sk_model.fit(
    X_train,
    y_train
)

end = time.perf_counter()

sk_train_time = end - start

#prediction(sklearn)

start = time.perf_counter()

sk_predictions = sk_model.predict(X_test)

end = time.perf_counter()

sk_prediction_time = end - start

#accuracy(sklearn)

sk_accuracy = np.mean(
    sk_predictions == y_test
)

print(f"Accuracy : {sk_accuracy*100:.2f}%")

print(f"Prediction Time : {sk_prediction_time:.6f} sec")

#confusion matrix(sklearn)

cm_sklearn = confusion_matrix(
    y_test,
    sk_predictions
)

#comparision

print("\nComparison\n")

print(f"scratch Accuracy : {scratch_accuracy*100:.2f}%")
print(f"Sklearn Accuracy : {sk_accuracy*100:.2f}%")

print()

print(f"scratch Prediction Time : {scratch_prediction_time:.6f}")
print(f"Sklearn Prediction Time : {sk_prediction_time:.6f}")

#print(X_test.shape)

