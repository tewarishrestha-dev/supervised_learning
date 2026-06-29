import time
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from model import SVM
from plots import (
    plot_accuracy,
    plot_training_time,
    plot_prediction_time,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_training_loss,
    plot_margin_distribution
)

# Load Dataset

data = load_breast_cancer()

X = data.data
y = data.target

# Convert labels to {-1, +1} for our implementation

y_custom = np.where(y == 0, -1, 1)

# Train-Test Split

X_train, X_test, y_train_custom, y_test_custom = train_test_split(
    X,
    y_custom,
    test_size=0.3,
    random_state=7
)

# sklearn needs labels as {0,1}
_, _, y_train_sklearn, y_test_sklearn = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=7
)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM Using SGD(stochastic gradient descent)

print("SVM using SGD\n")

our_model = SVM(
    learning_rate=0.001,
    lambda_param=0.01,
    n_iters=1000
)

start = time.perf_counter()

our_model.fit(X_train, y_train_custom)

end = time.perf_counter()

our_train_time = end - start

plot_margin_distribution(
    our_model,
    X_train,
    y_train_custom
)

plot_training_loss(our_model.loss_history)

start = time.perf_counter()

our_predictions = our_model.predict(X_test)

end = time.perf_counter()

our_prediction_time = end - start

our_accuracy = np.mean(our_predictions == y_test_custom)

print(f"Accuracy        : {our_accuracy*100:.2f}%")
print(f"Training Time   : {our_train_time:.5f} sec")
print(f"Prediction Time : {our_prediction_time:.8f} sec\n")

# Convert {-1,+1} → {0,1} for confusion matrix

our_predictions_binary = np.where(our_predictions == -1, 0, 1)
y_test_binary = np.where(y_test_custom == -1, 0, 1)

cm_ours = confusion_matrix(
    y_test_binary,
    our_predictions_binary
)

# Sklearn SVM

print("SCIKIT-LEARN SVM\n")

sk_model = SVC(
    kernel="linear",
    C=1.0
)

start = time.perf_counter()

sk_model.fit(X_train, y_train_sklearn)

end = time.perf_counter()

sk_train_time = end - start

start = time.perf_counter()

sk_predictions = sk_model.predict(X_test)

end = time.perf_counter()

sk_prediction_time = end - start

cm_sklearn = confusion_matrix(
    y_test_sklearn,
    sk_predictions
)

sk_accuracy = np.mean(sk_predictions == y_test_sklearn)

print(f"Accuracy        : {sk_accuracy*100:.2f}%")
print(f"Training Time   : {sk_train_time:.5f} sec")
print(f"Prediction Time : {sk_prediction_time:.8f} sec\n")

# Comparison

print("Comparison\n")

print(f"SVM using SGD Accuracy : {our_accuracy*100:.2f}%")
print(f"Sklearn Accuracy       : {sk_accuracy*100:.2f}%")

print()

print(f"SVM using SGD train time : {our_train_time:.5f} sec")
print(f"Sklearn Train Time       : {sk_train_time:.5f} sec")

print()

print(f"SVM using SGD predict time : {our_prediction_time:.8f} sec")
print(f"Sklearn Predict Time       : {sk_prediction_time:.8f} sec")

#plots

plot_accuracy(
    our_accuracy,
    sk_accuracy
)

plot_training_time(
    our_train_time,
    sk_train_time
)

plot_prediction_time(
    our_prediction_time,
    sk_prediction_time
)

plot_confusion_matrix(
    cm_ours,
    "Our SVM Confusion Matrix",
    "confusion_matrix_ours.png"
)

plot_confusion_matrix(
    cm_sklearn,
    "Sklearn SVM Confusion Matrix",
    "confusion_matrix_sklearn.png"
)

plot_feature_importance(
    our_model.w,
    data.feature_names
)

