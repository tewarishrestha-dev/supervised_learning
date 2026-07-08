import os
import numpy as np
import matplotlib.pyplot as plt
from model import KNN
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
os.makedirs("images", exist_ok=True)

data = load_breast_cancer()

X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=7
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)    

metrics = [
    "euclidean",
    "manhattan",
    "minkowski",
    "chebyshev"
]
accuracies = []

for metric in metrics:

    model = KNN(
        k=4,
        distance=metric,
        p=3
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = np.mean(
        predictions == y_test
    )

    accuracies.append(
        accuracy
    )
    print(
        f"{metric}: {accuracy*100:.2f}%"
    )

plt.figure(figsize=(8,5))


plt.bar(
    metrics,
    np.array(accuracies)*100
)


plt.xlabel(
    "Distance Metric"
)


plt.ylabel(
    "Accuracy (%)"
)


plt.title(
    "KNN Accuracy Comparison for Distance Metrics"
)


for i, value in enumerate(accuracies):

    plt.text(
        i,
        value*100 + 0.5,
        f"{value*100:.2f}%"
    )

plt.ylim(0,100)
plt.grid(axis="y")
plt.savefig(
    "images/distance_metric_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()