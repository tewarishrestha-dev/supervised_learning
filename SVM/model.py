import numpy as np


class SVM:

    def __init__(
        self,
        learning_rate=0.001,
        lambda_param=0.01,
        n_iters=1000
    ):

        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

        self.w = None
        self.b = None

        self.loss_history = []

    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            loss = 0

            for idx, x_i in enumerate(X):

                margin = y[idx] * (np.dot(x_i, self.w) + self.b)

                hinge_loss = max(0, 1 - margin)

                loss += hinge_loss

            regularization = 0.5 * self.lambda_param * np.dot(self.w, self.w)

            total_loss = regularization + loss

            self.loss_history.append(total_loss)

            for idx, x_i in enumerate(X):

                # Compute margin
                margin = y[idx] * (np.dot(x_i, self.w) + self.b)

                # Check if sample satisfies margin constraint
                if margin >= 1:

                    # Only regularization term contributes
                    self.w -= self.lr * (self.lambda_param * self.w)

                else:

                    # Hinge loss + regularization
                    self.w -= self.lr * (
                        self.lambda_param * self.w
                        - y[idx] * x_i
                    )

                    self.b += self.lr * y[idx]

    def predict(self, X):

        linear_output = np.dot(X, self.w) + self.b

        predictions = np.where(linear_output >= 0, 1, -1)

        return predictions




