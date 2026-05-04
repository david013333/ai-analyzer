import numpy as np


# Perceptron implementation
class Perceptron:
    def __init__(self, learning_rate=0.1, n_epochs=10):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}")
            for i, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else 0
                # Update rule
                update = self.lr * (y[i] - y_pred)
                self.weights += update * x_i
                self.bias += update
                print(f" Sample {x_i}, Target={y[i]}, Pred={y_pred}, Weights={self.weights},Bias = {self.bias}")


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
# Define datasets
OR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
OR_y = np.array([0, 1, 1, 1])
AND_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
AND_y = np.array([0, 0, 0, 1])
XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_y = np.array([0, 1, 1, 0])
# Train on OR
print("\nTraining on OR dataset:")
perceptron_or = Perceptron(learning_rate=0.1, n_epochs=5)
perceptron_or.fit(OR_X, OR_y)
print("Predictions:", perceptron_or.predict(OR_X))
# Train on AND
print("\nTraining on AND dataset:")
perceptron_and = Perceptron(learning_rate=0.1, n_epochs=5)
perceptron_and.fit(AND_X, AND_y)
print("Predictions:", perceptron_and.predict(AND_X))
# Train on XOR
print("\nTraining on XOR dataset:")
perceptron_xor = Perceptron(learning_rate=0.1, n_epochs=5)
perceptron_xor.fit(XOR_X, XOR_y)
print("Predictions:", perceptron_xor.predict(XOR_X))