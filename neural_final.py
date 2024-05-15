import numpy as np
import pandas as pd
from neural_lib import*

data_train = pd.read_csv(r".\data\ds2_train.csv")
data_test = pd.read_csv(r".\data\ds2_test.csv")
X_train = data_train.iloc[:, :-1].values.T
Y_train = data_train.iloc[:, -1].values.reshape(1, -1)
X_test = data_test.iloc[:, :-1].values.T
Y_test = data_test.iloc[:, -1].values.reshape(1, -1)
m,n = X_train.shape

learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1]
iterations = [100, 500, 1000, 1500, 2000]

best_params = hyperparameter_tuning(X_train, Y_train, X_test, Y_test, learning_rates, iterations,m)
print("Best Hyperparameters:", best_params)
