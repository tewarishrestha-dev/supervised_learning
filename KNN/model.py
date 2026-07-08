import numpy as np
class KNN:
    def __init__(
    self,
    k=5,
    distance="euclidean",
    weighted=False,
    p=2
):

        self.k = k
        self.distance = distance
        self.weighted = weighted
        self.p = p

        self.X_train = None
        self.y_train = None

    def fit(self, X, y):

       self.X_train = X
       self.y_train = y   

    def euclidean_distance(self, x1, x2):
       distance = np.sqrt(np.sum((x1 - x2) ** 2))
       return distance

    def manhattan_distance(self, x1, x2):
        distance = np.sum(np.abs(x1 - x2))
        return distance

    def minkowski_distance(self, x1, x2):
        distance = np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
        return distance

    def chebyshev_distance(self, x1, x2):
        distance = np.max(np.abs(x1 - x2))
        return distance

    def calculate_distance(self, x1, x2):

        if self.distance == "euclidean":
            return self.euclidean_distance(x1, x2)

        elif self.distance == "manhattan":
            return self.manhattan_distance(x1, x2)

        elif self.distance == "minkowski":
            return self.minkowski_distance(x1, x2)

        elif self.distance == "chebyshev":
            return self.chebyshev_distance(x1, x2)

        else:
            raise ValueError("Invalid distance metric")
    
    def _predict(self, x):
        distances = []

        for x_train in self.X_train:
            distance = self.calculate_distance(x, x_train)
            distances.append(distance)  

        sorted_indices = np.argsort(distances)
        k_indices = sorted_indices[:self.k]
        k_distances = [
            distances[i]
            for i in k_indices
        ]  
        k_nearest_labels = [self.y_train[i] for i in k_indices]  
        if self.weighted:
            class_votes = {}
            for label, distance in zip(
                k_nearest_labels,
                k_distances
            ):
                weight = 1 / (distance + 1e-10)
                if label not in class_votes:
                    class_votes[label] = 0
                class_votes[label] += weight
            prediction = max(
                class_votes,
                key=class_votes.get
            )
        else:
            labels, counts = np.unique(
                k_nearest_labels,
                return_counts=True
            )
            prediction = labels[np.argmax(counts)]  
        return prediction
    

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)     
    