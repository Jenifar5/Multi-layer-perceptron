import numpy as np
def relu(x):
    return np.maximum(0, x)
def tanh(x):
    return np.tanh(x)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
def cross_entropy(predictions, labels):
    m = labels.shape[0]
    log_likelihood = -np.log(predictions[range(m), labels])
    return np.sum(log_likelihood) / m
def grad_cross_entropy(predictions, labels):
    m = labels.shape[0]
    grad = predictions.copy()
    grad[range(m), labels] -= 1
    grad /= m
    return grad
class MLP:
    def __init__(self, input_size=2, hidden1=8, hidden2=6, output_size=3, lr=0.5):
        self.W1 = np.random.randn(input_size, hidden1) * 0.01
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * 0.01
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))
        self.lr = lr
    def forward(self, X):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = tanh(self.z2)
        self.z3 = self.a2.dot(self.W3) + self.b3
        self.a3 = softmax(self.z3)
        return self.a3
    def backward(self, X, y, output):
        m = X.shape[0]
        grad_output = grad_cross_entropy(output, y)
        dW3 = self.a2.T.dot(grad_output)
        db3 = np.sum(grad_output, axis=0, keepdims=True)
        grad_a2 = grad_output.dot(self.W3.T)
        grad_z2 = grad_a2 * (1 - np.tanh(self.z2)**2)
        dW2 = self.a1.T.dot(grad_z2)
        db2 = np.sum(grad_z2, axis=0, keepdims=True)
        grad_a1 = grad_z2.dot(self.W2.T)
        grad_z1 = grad_a1 * (self.z1 > 0)
        dW1 = X.T.dot(grad_z1)
        db1 = np.sum(grad_z1, axis=0, keepdims=True)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
    def train(self, X, y, epochs=20):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = cross_entropy(output, y)
            self.backward(X, y, output)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1), probs
X = np.array([[0.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 1.0]])
y = np.array([0, 1, 2, 1])
model = MLP()
model.train(X, y, epochs=20)
test_input = np.array([[0.5, 0.5]])
pred_class, pred_probs = model.predict(test_input)
print("Prediction probabilities:", pred_probs)
print("Predicted class:", pred_class[0])