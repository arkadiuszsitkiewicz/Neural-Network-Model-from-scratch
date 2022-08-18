import numpy as np
from src.NN_model.activation_funcs import sigmoid, relu, tanh, softmax
from src.NN_model.activation_funcs import sigmoid_backward, relu_backward, tanh_backward, softmax_backward


class NeuralNetworkModel:
    __ACTIVATION_FUNC = ("sigmoid", "relu", "tanh")
    __OPTIMIZERS = ('gd', "momentum", "rms_prop", "adam")

    @classmethod
    def get_activators(cls):
        return cls.__ACTIVATION_FUNC

    @classmethod
    def get_optimizers(cls):
        return cls.__OPTIMIZERS

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
        elif activation == 'softmax':
            A, activation_cache = softmax(Z)

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
        # last layer always sigmoid or softmax activation
        last_activation = 'softmax' if params["b" + str(L)].shape[0] > 1 else 'sigmoid'
        AL, cache = NeuralNetworkModel.__linear_activation_forward(A, params["W" + str(L)], params["b" + str(L)],
                                                                   activation=last_activation,
                                                                   dropout_keep_prob=dropout_keep_prob)
        caches.append(cache)
        return AL, caches

    @staticmethod
    def __compute_cost(AL, Y, params, l2_lambda):
        if Y.shape[0] > 1:
            cost = - np.sum(Y * np.log(AL + 1e-15))
            cost = np.squeeze(cost)
        else:
            cost = - (np.dot(Y, np.log(AL.T + 1e-15)) + np.dot(1 - Y, np.log(1 - AL + 1e-15).T))
            cost = np.squeeze(cost)

        if l2_lambda != 0:
            L = len(params) // 2
            l2_regularization_cost = 0
            for i in range(L):
                l2_regularization_cost += np.sum(np.square(params["W" + str(i + 1)]))
            cost += np.squeeze(l2_regularization_cost) * l2_lambda / 2
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
    def __linear_activation_backward(dA, cache, activation, l2_lambda, dropout_keep_prob, mask=None):
        linear_cache, activation_cache = cache

        if activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
        elif activation == "relu":
            dZ = relu_backward(dA, activation_cache)
        elif activation == "tanh":
            dZ = tanh_backward(dA, activation_cache)
        elif activation == "softmax":
            dZ = softmax_backward(dA, activation_cache, mask)

        dA_prev, dW, db = NeuralNetworkModel.__linear_backward(dZ, linear_cache, l2_lambda=l2_lambda,
                                                               dropout_keep_prob=dropout_keep_prob)
        return dA_prev, dW, db

    @staticmethod
    def __l_model_backward(AL, Y, caches, activation, l2_lambda, dropout_keep_prob, mask):
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)
        if Y.shape[0] > 1:
            dAL = - np.divide(Y, AL + 1e-15)
            last_activation = 'softmax'
        else:
            dAL = - np.divide(Y, AL + 1e-15) + np.divide(1 - Y, 1 - AL + 1e-15)
            last_activation = 'sigmoid'
        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = NeuralNetworkModel.__linear_activation_backward(dAL, current_cache,
                                                                                         last_activation,
                                                                                         l2_lambda=l2_lambda,
                                                                                         dropout_keep_prob=dropout_keep_prob,
                                                                                         mask=mask)
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
                                                                                         current_cache, activation,
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

        for l in range(1, L + 1):
            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        return parameters

    @staticmethod
    def __random_mini_batches(X, Y, mini_batch_size=64):
        m = X.shape[1]
        mini_batches = []
        permutation = list(np.random.permutation(m))
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation].reshape((-1, m))

        num_minibatches = np.ceil(m / mini_batch_size).astype(int)
        for i in range(num_minibatches):
            mini_batch_X = X_shuffled[:, i*mini_batch_size:(i + 1)*mini_batch_size]
            mini_batch_Y = Y_shuffled[:, i*mini_batch_size:(i + 1)*mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches

    @staticmethod
    def __initialize_momentum_or_rms_prop(params):
        L = len(params) // 2
        v = {}
        for l in range(1, L+1):
            v["dW" + str(l)] = np.zeros((params["W" + str(l)].shape[0], params["W" + str(l)].shape[1]))
            v["db" + str(l)] = np.zeros((params["b" + str(l)].shape[0], params["b" + str(l)].shape[1]))
        return v

    @staticmethod
    def __update_params_momentum(params, grads, v, beta1, learning_rate):
        L = len(params) // 2
        for l in range(1, L + 1):
            v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
            v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
            params["W" + str(l)] = params["W" + str(l)] - learning_rate * v["dW" + str(l)]
            params["b" + str(l)] = params["b" + str(l)] - learning_rate * v["db" + str(l)]
        return params, v

    @staticmethod
    def __update_params_rms_prop(params, grads, s, beta2, learning_rate, epsilon):
        L = len(params) // 2
        for l in range(1, L + 1):
            s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
            s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)
            params["W" + str(l)] = params["W" + str(l)] - learning_rate * grads["dW" + str(l)] / (
                        np.sqrt(s["dW" + str(l)]) + epsilon)
            params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["db" + str(l)] / (
                        np.sqrt(s["db" + str(l)]) + epsilon)
        return params, s

    @staticmethod
    def __initialize_adam(params):
        L = len(params) // 2
        v = {}
        s = {}
        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros((params["W" + str(l)].shape[0], params["W" + str(l)].shape[1]))
            v["db" + str(l)] = np.zeros((params["b" + str(l)].shape[0], params["b" + str(l)].shape[1]))
            s["dW" + str(l)] = np.zeros((params["W" + str(l)].shape[0], params["W" + str(l)].shape[1]))
            s["db" + str(l)] = np.zeros((params["b" + str(l)].shape[0], params["b" + str(l)].shape[1]))
        return v, s

    @staticmethod
    def __update_params_adam(params, grads, v, s, t, learning_rate, beta1, beta2, epsilon):
        L = len(params) // 2
        v_corrected = {}
        s_corrected = {}
        for l in range(1, L + 1):
            v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
            v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)
            s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
            s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)
            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
            s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)
            params["W" + str(l)] = params["W" + str(l)] - learning_rate * v_corrected["dW" + str(l)] / (
                        np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
            params["b" + str(l)] = params["b" + str(l)] - learning_rate * v_corrected["db" + str(l)] / (
                        np.sqrt(s_corrected["db" + str(l)]) + epsilon)
        return params, v, s

    @staticmethod
    def __schedule_l_rate_decay(l_rate0, epoch_num, decay_rate, time_interval):
        return l_rate0 / (1 + decay_rate * np.floor(epoch_num / time_interval))

    @staticmethod
    def __create_masks(c, m, mini_batch_size):
        masks = []
        num_full_mini_batches = m // mini_batch_size
        mask = np.zeros((c * mini_batch_size, c * mini_batch_size))
        for i in range(0, c * mini_batch_size, c):
            mask[i:i+c, i:i+c] = 1
        for i in range(num_full_mini_batches):
            masks.append(mask)
        if m % mini_batch_size != 0:
            last_mini_batch_size = m % mini_batch_size
            last_mask = np.zeros((c * last_mini_batch_size, c * last_mini_batch_size))
            for i in range(0, c * last_mini_batch_size, c):
                last_mask[i:i + c, i:i + c] = 1
            masks.append(last_mask)
        return masks

    def __init__(self, hidden_layer_dims, activation="relu", optimizer='gd', learning_rate=0.01, mini_batch_size=64,
                 beta1=0.9, beta2=0.999, epsilon=1e-8, decay=None, decay_rate=1, time_interval=1000,
                 num_epochs=2500, l2_lambda=0, dropout_keep_prob=1, save_costs=0, print_cost=0):
        if type(hidden_layer_dims) != list:
            raise TypeError(f"{hidden_layer_dims} must be a list type object")
        else:
            self.hidden_layer_dims = hidden_layer_dims
        if activation not in NeuralNetworkModel.__ACTIVATION_FUNC:
            raise ValueError(f"{activation} is not a valid activation function")
        else:
            self.activation = activation
        if optimizer not in NeuralNetworkModel.__OPTIMIZERS:
            raise ValueError(f"{optimizer} is not a valid activation function")
        else:
            self.optimizer = optimizer
        self.mini_batch_size = mini_batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.decay_rate = decay_rate
        self.time_interval = time_interval
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
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
        c, m = Y.shape
        learning_rate = self.learning_rate
        costs = [] if self.save_costs else None
        mini_batch_size = m if self.mini_batch_size == 0 else self.mini_batch_size
        masks = NeuralNetworkModel.__create_masks(c, m, mini_batch_size)

        params = self.__initialize_params(X.shape[0], Y.shape[0]) if self.__params is None else self.__params
        if self.optimizer == 'gd':
            pass
        elif self.optimizer == 'momentum':
            v = NeuralNetworkModel.__initialize_momentum_or_rms_prop(params)
        elif self.optimizer == 'rms_prop':
            s = NeuralNetworkModel.__initialize_momentum_or_rms_prop(params)
        elif self.optimizer == 'adam':
            t = 0
            v, s = NeuralNetworkModel.__initialize_adam(params)

        for i in range(self.num_epochs):
            minibatches = NeuralNetworkModel.__random_mini_batches(X, Y, mini_batch_size)
            cost_total = 0
            for minibatch, mask in zip(minibatches, masks):
                (mini_batch_X, mini_batch_Y) = minibatch

                AL, caches = NeuralNetworkModel.__l_model_forward(X=mini_batch_X, params=params,
                                                                  activation=self.activation,
                                                                  dropout_keep_prob=self.dropout_keep_prob)
                cost_total += NeuralNetworkModel.__compute_cost(AL=AL, Y=mini_batch_Y, params=params,
                                                                l2_lambda=self.l2_lambda)
                grads = NeuralNetworkModel.__l_model_backward(AL=AL, Y=mini_batch_Y, caches=caches,
                                                              activation=self.activation, l2_lambda=self.l2_lambda,
                                                              dropout_keep_prob=self.dropout_keep_prob,
                                                              mask=mask)
                if self.optimizer == 'gd':
                    params = NeuralNetworkModel.__update_params(params=params, grads=grads,
                                                                learning_rate=learning_rate)
                elif self.optimizer == 'momentum':
                    params, v = NeuralNetworkModel.__update_params_momentum(params=params, grads=grads, v=v,
                                                                            beta1=self.beta1,
                                                                            learning_rate=learning_rate)
                elif self.optimizer == 'rms_prop':
                    params, s = NeuralNetworkModel.__update_params_rms_prop(params=params, grads=grads, s=s,
                                                                            beta2=self.beta2, epsilon=self.epsilon,
                                                                            learning_rate=self.learning_rate)
                elif self.optimizer == 'adam':
                    t += 1
                    params, v, s = NeuralNetworkModel.__update_params_adam(params=params, grads=grads, v=v, s=s, t=t,
                                                                           learning_rate=learning_rate,
                                                                           beta1=self.beta1, beta2=self.beta2,
                                                                           epsilon=self.epsilon)
            cost_total /= m
            if costs is not None and (i % self.save_costs == 0 or i == self.num_epochs - 1):
                costs.append(cost_total)
            if (self.print_cost and i % self.print_cost == 0) or i == self.num_epochs - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost_total)))
            if self.decay and i % self.time_interval == 0:
                learning_rate = NeuralNetworkModel.__schedule_l_rate_decay(self.learning_rate, epoch_num=i,
                                                                           decay_rate=self.decay_rate,
                                                                           time_interval=self.time_interval)
        self.__params = params
        self.__costs = costs

    def predict(self, X):
        if self.__params is None:
            raise ValueError(f"Parameters are not provided. Fit model or upload parameters")
        probas, _ = NeuralNetworkModel.__l_model_forward(X, self.__params, self.activation, dropout_keep_prob=1)
        if probas.shape[0] == 1:
            p = (probas > 0.5).astype(int)
        else:
            idx = np.argmax(probas, axis=0)
            p = np.zeros_like(probas)
            p[idx, np.arange(len(idx))] = 1
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
        print(f"Number of output_classes: {2 if output_classes == 1 else output_classes}")
        print(f"Classification method: {'binary' if output_classes == 1 else 'multiclass-classification'}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Mini-batch size {self.mini_batch_size}")
        print(f"Activation function: {self.activation}")
        print(f"Optimizer: {self.optimizer}")
        if self.optimizer == 'momentum':
            print(f"beta1 for momentum {self.beta1}")
        elif self.optimizer == 'rms_prop':
            print(f"beta2 for rms_prop {self.beta2}")
            print(f"Epsilon for rms_prop: {self.epsilon}")
        elif self.optimizer == "adam":
            print(f"beta1, beta2 for adam {self.beta1}, {self.beta2}")
            print(f"Epsilon for adam: {self.epsilon}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Learning rate decay: {self.decay}")
        if self.decay:
            print(f"Decay rate: {self.decay_rate}", f"Time interval: {self.time_interval}", sep="\n")
        print(f"L2: {'no' if self.l2_lambda == 0 else 'lambda value ' + str(self.l2_lambda)}")
        print(f"Dropout: {'no' if self.dropout_keep_prob == 1 else 'keep_prob ' + str(self.dropout_keep_prob)}")
        print(f"Last cost: {self.__costs if self.__costs is None else np.round(self.__costs[-1], 4)}")
