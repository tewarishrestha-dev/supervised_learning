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

model = KNN(
    k=4,
    distance="euclidean"
)

model.fit(
    X_train,
    y_train
)

all_distances = []
for x in X_test:

    distances = []

    for x_train in X_train:

        distance = np.sqrt(
            np.sum(
                (x-x_train)**2
            )
        )

        distances.append(distance)
    distances = np.sort(distances)
    nearest_distances = distances[:4]
    all_distances.extend(
        nearest_distances
    )
plt.figure(figsize=(8,5))
plt.hist(
    all_distances,
    bins=30
)
plt.xlabel(
    "Distance to Neighbor"
)
plt.ylabel(
    "Frequency"
)
plt.title(
    "Distribution of K Nearest Neighbor Distances"
)
plt.grid(True)
plt.savefig(
    "images/neighbor_distance_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()    