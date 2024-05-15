import numpy as np
import pandas as pd
def init_params():
    np.random.seed(42)
    W1 = np.random.rand(2, 2) - 0.5
    b1 = np.random.rand(2, 1) - 0.5
    W2 = np.random.rand(2, 2) - 0.5
    b2 = np.random.rand(2, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    Y = Y.astype(int)
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def accuracy(y_pred, y_test):
    return np.mean(y_pred == y_test)

def hyperparameter_tuning(X_train, Y_train, X_test, Y_test, learning_rates, iterations, p_m):
    global m
    m = p_m
    best_accuracy = 0
    best_params = {}
    
    for alpha in learning_rates:
        for num_iterations in iterations:
            W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha, num_iterations)
            y_pred = get_predictions(forward_prop(W1, b1, W2, b2, X_test)[-1])
            acc_score = accuracy(y_pred, Y_test)  # Rename the variable here
            
            print(f"Learning Rate: {alpha}, Iterations: {num_iterations}, Accuracy: {acc_score}")
            
            if acc_score > best_accuracy:
                best_accuracy = acc_score
                best_params = {'learning_rate': alpha, 'iterations': num_iterations}
    print("Best accuracy ",best_accuracy)
    return best_params