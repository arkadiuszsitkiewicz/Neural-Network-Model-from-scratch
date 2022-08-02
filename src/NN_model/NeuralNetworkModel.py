import numpy as np

from src.NN_model.math_calc import *


class NeuralNetworkModel:
    __ACTIVATION_FUNC = ("sigmoid", "relu", "tanh")

    @classmethod
    def get_activators(cls):
        return cls.__ACTIVATION_FUNC

    @staticmethod
    def __linear_forward(A, W, b, dropout_keep_prob):
        if dropout_keep_prob == 1:
            cache = (A, W, b)
        else:
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < dropout_keep_prob).astype(int)
            A = A * D
            A = A / dropout_keep_prob
            cache = (A, D, W, b)
        Z = np.dot(W, A) + b
        return Z, cache

    @staticmethod
    def __linear_activation_forward(A_prev, W, b, activation, dropout_keep_prob):
        Z, linear_cache = NeuralNetworkModel.__linear_forward(A_prev, W, b, dropout_keep_prob=dropout_keep_prob)
        if activation == "sigmoid":
            A, activation_cache = sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = relu(Z)
        elif activation == "tanh":
            A, activation_cache = tanh(Z)

        cache = (linear_cache, activation_cache)
        return A, cache

    @staticmethod
    def __l_model_forward(X, params, activation, dropout_keep_prob):
        L = len(params) // 2
        caches = []
        # X always without dropout
        A, cache = NeuralNetworkModel.__linear_activation_forward(X, params["W" + str(1)], params["b" + str(1)],
                                                                  activation, dropout_keep_prob=1)
        caches.append(cache)

        for l in range(2, L):
            A_prev = A
            A, cache = NeuralNetworkModel.__linear_activation_forward(A_prev, params["W" + str(l)],
                                                                      params["b" + str(l)],
                                                                      activation, dropout_keep_prob=dropout_keep_prob)
            caches.append(cache)
        # last layer always with sigmoid activation
        AL, cache = NeuralNetworkModel.__linear_activation_forward(A, params["W" + str(L)], params["b" + str(L)],
                                                                   "sigmoid",
                                                                   dropout_keep_prob=dropout_keep_prob)
        caches.append(cache)
        return AL, caches

    @staticmethod
    def __compute_cost(AL, Y, params, l2_lambda):
        m = Y.shape[1]
        cost = - 1 / m * (np.dot(Y, np.log(AL.T + 1e-15)) + np.dot(1 - Y, np.log(1 - AL + 1e-15).T))
        cost = np.squeeze(cost)
        if l2_lambda != 0:
            L = len(params) // 2
            l2_regularization_cost = 0
            for i in range(L):
                l2_regularization_cost += np.sum(np.square(params["W" + str(i + 1)]))
            cost += np.squeeze(l2_regularization_cost) * l2_lambda / (2 * m)
        return cost

    @staticmethod
    def __linear_backward(dZ, cache, l2_lambda, dropout_keep_prob):
        if dropout_keep_prob == 1:
            A_prev, W, b = cache
        else:
            A_prev, D_prev, W, b = cache

        m = A_prev.shape[1]
        dW = 1 / m * np.dot(dZ, A_prev.T)
        if l2_lambda != 0:
            dW = dW + (l2_lambda / m * W)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        dA_prev = np.dot(W.T, dZ)
        if dropout_keep_prob != 1:
            dA_prev = dA_prev * D_prev
            dA_prev = dA_prev / dropout_keep_prob
        return dA_prev, dW, db

    @staticmethod
    def __linear_activation_backward(dA, cache, activation, l2_lambda, dropout_keep_prob):
        linear_cache, activation_cache = cache

        if activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
        elif activation == "relu":
            dZ = relu_backward(dA, activation_cache)
        elif activation == "tanh":
            dZ = tanh_backward(dA, activation_cache)

        dA_prev, dW, db = NeuralNetworkModel.__linear_backward(dZ, linear_cache, l2_lambda=l2_lambda,
                                                               dropout_keep_prob=dropout_keep_prob)
        return dA_prev, dW, db

    @staticmethod
    def __l_model_backward(AL, Y, caches, activation, l2_lambda, dropout_keep_prob):
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)

        dAL = - np.divide(Y, AL + 1e-15) + np.divide(1 - Y, 1 - AL + 1e-15)
        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = NeuralNetworkModel.__linear_activation_backward(dAL, current_cache, "sigmoid",
                                                                                         l2_lambda=l2_lambda,
                                                                                         dropout_keep_prob=dropout_keep_prob)
        grads["dA" + str(L - 1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        for l in reversed(range(1, L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = NeuralNetworkModel.__linear_activation_backward(grads["dA" + str(l + 1)],
                                                                                             current_cache, activation,
                                                                                             l2_lambda=l2_lambda,
                                                                                             dropout_keep_prob=dropout_keep_prob)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        current_cache = caches[0]
        dA_prev_temp, dW_temp, db_temp = NeuralNetworkModel.__linear_activation_backward(grads["dA" + str(1)],
                                                                                         current_cache, "sigmoid",
                                                                                         l2_lambda=l2_lambda,
                                                                                         dropout_keep_prob=1)
        grads["dA" + str(0)] = dA_prev_temp
        grads["dW" + str(1)] = dW_temp
        grads["db" + str(1)] = db_temp
        return grads

    @staticmethod
    def __update_params(params, grads, learning_rate):
        parameters = params.copy()
        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        return parameters

    def __init__(self, hidden_layer_dims, activation="relu", learning_rate=0.01,
                 num_iteration=2500, l2_lambda=0, dropout_keep_prob=1, save_costs=0, print_cost=0):
        if type(hidden_layer_dims) != list:
            raise TypeError(f"{hidden_layer_dims} must be a list type object")
        else:
            self.hidden_layer_dims = hidden_layer_dims
        if activation not in NeuralNetworkModel.__ACTIVATION_FUNC:
            raise ValueError(f"{activation} is not a valid activation function")
        else:
            self.activation = activation
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.l2_lambda = l2_lambda
        self.dropout_keep_prob = dropout_keep_prob
        self.save_costs = save_costs
        self.print_cost = print_cost
        self.__params = None
        self.__costs = None

    def __initialize_params(self, n_x, n_y):
        params = {}
        layers = [n_x] + self.hidden_layer_dims + [n_y]
        L = len(layers)

        if self.activation == "relu":
            const = 2
        elif self.activation == "tanh" or self.activation == "sigmoid":
            const = 1
        for i in range(1, L):
            params["W" + str(i)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(const / layers[i - 1])
            params["b" + str(i)] = np.zeros((layers[i], 1))
        return params

    def fit(self, X, Y):
        costs = [] if self.save_costs else None
        params = self.__initialize_params(X.shape[0], Y.shape[0]) if self.__params is None else self.__params

        for i in range(self.num_iteration):
            AL, caches = NeuralNetworkModel.__l_model_forward(X=X, params=params, activation=self.activation,
                                                              dropout_keep_prob=self.dropout_keep_prob)
            cost = NeuralNetworkModel.__compute_cost(AL=AL, Y=Y, params=params, l2_lambda=self.l2_lambda)
            grads = NeuralNetworkModel.__l_model_backward(AL=AL, Y=Y, caches=caches, activation=self.activation,
                                                          l2_lambda=self.l2_lambda,
                                                          dropout_keep_prob=self.dropout_keep_prob)
            params = NeuralNetworkModel.__update_params(params=params, grads=grads, learning_rate=self.learning_rate)
            if costs is not None and (i % self.save_costs == 0 or i == self.num_iteration - 1):
                costs.append(cost)
            if (self.print_cost and i % self.print_cost == 0) or i == self.num_iteration - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        self.__params = params
        self.__costs = costs

    def predict(self, X):
        if self.__params is None:
            raise ValueError(f"Parameters are not provided. Fit model or upload parameters")
        m = X.shape[1]
        p = np.zeros((1, m))
        probas, _ = NeuralNetworkModel.__l_model_forward(X, self.__params, self.activation, dropout_keep_prob=1)
        p[probas > 0.5] = 1
        return p

    def predict_proba(self, X):
        if self.__params is None:
            raise ValueError(f"Parameters are not provided. Fit model or upload parameters")
        probas, _ = NeuralNetworkModel.__l_model_forward(X, self.__params, self.activation, dropout_keep_prob=1)
        return probas

    def upload_params(self, params):
        self.__params = params

    def get_params(self):
        return self.__params

    def get_costs(self):
        return self.__costs

    def get_info(self):
        print(f"Hidden layer dimensions: {self.hidden_layer_dims}")
        if self.__params is not None:
            input_features = self.__params['W1'].shape[1]
            output_classes = self.__params['W' + str(len(self.hidden_layer_dims) + 1)].shape[0]
        else:
            input_features = 'not specified (i.e. model not trained)'
            output_classes = 'not specified (i.e. model not trained)'
        print(f"Input features: {input_features}")
        print(f"Number of output_classes: {output_classes}")
        print(f"Activation function: {self.activation}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Iterations: {self.num_iteration}")
        print(f"L2: {'no' if self.l2_lambda == 0 else 'lambda value ' + str(self.l2_lambda)}")
        print(f"Dropout: {'no' if self.dropout_keep_prob == 1 else 'keep_prob ' + str(self.dropout_keep_prob)}")
        print(f"Last cost: {self.__costs if self.__costs is None else np.round(self.__costs[-1], 2)}")
