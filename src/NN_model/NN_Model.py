from src.NN_model.math_calc import *


class NN_Model():
    __SOLVER_TYPES = ("sigmoid", "relu", "tanh")

    @classmethod
    def get_solvers(cls):
        return cls.__SOLVER_TYPES

    @staticmethod
    def __linear_forward(A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    @staticmethod
    def __linear_activation_forward(A_prev, W, b, activation):
        Z, linear_cache = NN_Model.__linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A, activation_cache = sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = relu(Z)
        elif activation == "tanh":
            A, activation_cache = tanh(Z)

        cache = (linear_cache, activation_cache)
        return A, cache

    @staticmethod
    def __l_model_forward(X, params, activation):
        A = X
        L = len(params) // 2
        caches = []
        for l in range(1, L):
            A_prev = A
            A, cache = NN_Model.__linear_activation_forward(A_prev, params["W" + str(l)],
                                                            params["b" + str(l)], activation)
            caches.append(cache)

        AL, cache = NN_Model.__linear_activation_forward(A, params["W" + str(L)],
                                                         params["b" + str(L)], "sigmoid")
        caches.append(cache)
        return AL, caches

    @staticmethod
    def __compute_cost(AL, Y):
        m = Y.shape[1]
        cost = - 1 / m * (np.dot(Y, np.log(AL.T)) + np.dot(1 - Y, np.log(1 - AL).T))
        return np.squeeze(cost)

    @staticmethod
    def __linear_backward(dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    @staticmethod
    def __linear_activation_backward(dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
        elif activation == "relu":
            dZ = relu_backward(dA, activation_cache)
        elif activation == "tanh":
            dZ = tanh_backward(dA, activation_cache)

        dA_prev, dW, db = NN_Model.__linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    @staticmethod
    def __l_model_backward(AL, Y, caches, activation="relu"):
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)

        dAL = - np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = NN_Model.__linear_activation_backward(dAL, current_cache, "sigmoid")
        grads["dA" + str(L - 1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = NN_Model.__linear_activation_backward(grads["dA" + str(l + 1)],
                                                                                   current_cache, activation)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        return grads

    @staticmethod
    def __update_params(params, grads, learning_rate):
        parameters = params.copy()
        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        return parameters

    def __init__(self, hidden_layer_dims, solver="relu", learning_rate=0.0075,
                 num_iteration=2500, print_cost=True):
        if type(hidden_layer_dims) != list:
            raise TypeError(f"{hidden_layer_dims} must be a list type object")
        else:
            self.hidden_layer_dims = hidden_layer_dims
        if solver not in NN_Model.__SOLVER_TYPES:
            raise ValueError(f"{solver} is not a valid solver")
        else:
            self.solver = solver
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.print_cost = print_cost
        self.__params = None

    def __initialize_params(self, n_x):
        np.random.seed(1)
        params = {}
        layers = self.hidden_layer_dims
        params["W1"] = np.random.randn(layers[0], n_x) / np.sqrt(n_x)  # * 0.01
        params["b1"] = np.zeros((layers[0], 1))
        L = len(layers)
        for i in range(1, L):
            params["W" + str(i + 1)] = np.random.randn(layers[i], layers[i - 1]) / np.sqrt(layers[i - 1])
            params["b" + str(i + 1)] = np.zeros((layers[i], 1))
        return params

    def fit(self, X, Y):
        num_iter = self.num_iteration
        learning_rate = self.learning_rate
        solver = self.solver
        print_cost = self.print_cost
        costs = []
        params = self.__initialize_params(X.shape[0])

        for i in range(num_iter):
            AL, caches = NN_Model.__l_model_forward(X, params, solver)
            cost = NN_Model.__compute_cost(AL, Y)
            grads = NN_Model.__l_model_backward(AL, Y, caches, solver)
            params = NN_Model.__update_params(params, grads, learning_rate)
            if print_cost and (i % 100 == 0 or i == num_iter - 1):
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iter - 1:
                costs.append(cost)
        self.__params = params
        return params, costs

    def predict(self, X):
        if self.__params is None:
            raise ValueError(f"Parameters are not provided. Fit model or upload parameters")
        m = X.shape[1]
        p = np.zeros((1, m))
        probas, _ = NN_Model.__l_model_forward(X, self.__params, self.solver)
        p[probas > 0.5] = 1
        return p

    def predict_proba(self, X):
        if self.__params is None:
            raise ValueError(f"Parameters are not provided. Fit model or upload parameters")
        probas, _ = NN_Model.__l_model_forward(X, self.__params, self.solver)
        return probas

    def upload_params(self, params):
        self.__params = params
