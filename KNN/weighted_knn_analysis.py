import numpy as np
import matplotlib.pyplot as plt
from model import KNN
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
models = [
    ("Standard KNN",False),
    ("Weighted KNN",True)
]
accuracies=[]

for name, weighted in models:


    model = KNN(
        k=4,
        distance="euclidean",
        weighted=weighted
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = np.mean(
        predictions==y_test
    )

    accuracies.append(
        accuracy
    )
    print(
        name,
        accuracy*100
    )

plt.bar(
    ["Standard KNN","Weighted KNN"],
    np.array(accuracies)*100
)
plt.ylabel(
    "Accuracy (%)"
)
plt.title(
    "Standard vs Weighted KNN"
)
plt.savefig(
    "images/weighted_knn_comparison.png",
    dpi=300
)
plt.show()