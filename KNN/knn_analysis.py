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

k_values = range(1,31)

accuracies = []

for k in k_values:

    model = KNN(
        k=k,
        distance="euclidean"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = np.mean(
        predictions == y_test
    )

    accuracies.append(
        accuracy
    )

plt.figure(figsize=(8,5))


plt.plot(
    k_values,
    np.array(accuracies)*100,
    marker="o",
    linewidth=2
)


plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy (%)")

plt.title(
    "KNN Accuracy vs K"
)


plt.grid(True)

plt.savefig(
    "images/accuracy_vs_k.png",
    dpi=300,
    bbox_inches="tight"
)
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)

plt.scatter(
    best_k,
    best_accuracy*100,
    color="red",
    s=100,
    label=f"Best K={best_k}"
)
print(
    f"Best K: {best_k}"
)

print(
    f"Best Accuracy: {best_accuracy*100:.2f}%"
)

plt.legend()
plt.show()    