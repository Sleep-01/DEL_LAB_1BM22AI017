#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.random.seed(42)
n_features = 10
n_samples = 100
n_hidden = 64
n_classes = 2
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, n_classes, size=(n_samples, 1))
y_one_hot = np.zeros((n_samples, n_classes))
y_one_hot[np.arange(n_samples), y.flatten()] = 1
def initialize_parameters():
    W1 = np.random.randn(n_features, n_hidden) * 0.01
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, n_classes) * 0.01
    b2 = np.zeros((1, n_classes))
    return W1, b1, W2, b2
def forward(X, W1, b1, W2, b2):
    Z1 = X.dot(W1) + b1
    A1 = np.maximum(0, Z1)  # ReLU activation
    Z2 = A1.dot(W2) + b2
    exp_scores = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))  # Softmax for numerical stability
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return Z1, A1, Z2, probs
def compute_loss(probs, y):
    corect_logprobs = -np.log(probs[range(len(y)), y.flatten()])
    data_loss = np.sum(corect_logprobs) / len(y)
    return data_loss
def backward(X, y, Z1, A1, probs, W1, W2):
    delta3 = probs
    delta3[range(len(y)), y.flatten()] -= 1
    delta3 /= len(y)
    dW2 = A1.T.dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(W2.T)
    delta2[Z1 <= 0] = 0
    dW1 = X.T.dot(delta2)
    db1 = np.sum(delta2, axis=0, keepdims=True)
    return dW1, db1, dW2, db2
def gradient_descent(X, y, W1, b1, W2, b2, learning_rate=0.01, epochs=1000):
    for epoch in range(epochs):
        # Forward pass
        Z1, A1, Z2, probs = forward(X, W1, b1, W2, b2)
        loss = compute_loss(probs, y)
        dW1, db1, dW2, db2 = backward(X, y, Z1, A1, probs, W1, W2)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return W1, b1, W2, b2
def stochastic_gradient_descent(X, y, W1, b1, W2, b2, learning_rate=0.01, epochs=1000):
    for epoch in range(epochs):
        for i in range(len(X)):
            Xi = X[i:i+1]
            yi = y[i:i+1]
            Z1, A1, Z2, probs = forward(Xi, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward(Xi, yi, Z1, A1, probs, W1, W2)
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
        if epoch % 100 == 0:
            Z1, A1, Z2, probs = forward(X, W1, b1, W2, b2)
            loss = compute_loss(probs, y)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return W1, b1, W2, b2
W1, b1, W2, b2 = initialize_parameters()
print("Training with Gradient Descent")
W1_gd, b1_gd, W2_gd, b2_gd = gradient_descent(X, y, W1, b1, W2, b2, learning_rate=0.01, epochs=1000)
print("\nTraining with Stochastic Gradient Descent")
W1_sgd, b1_sgd, W2_sgd, b2_sgd = stochastic_gradient_descent(X, y, W1, b1, W2, b2, learning_rate=0.01, epochs=1000)


# In[ ]:




