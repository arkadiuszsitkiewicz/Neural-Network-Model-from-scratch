import numpy as np


class DataTransformation:
    @staticmethod
    def one_hot_matrix(y):
        num_classes = y.max() + 1
        y_one_hot = np.zeros((num_classes, y.shape[1]))
        y_one_hot[y, np.arange(0, y.shape[1])] = 1
        return y_one_hot

    @staticmethod
    def one_hot_to_num(one_hot_matrix):
        return np.argmax(one_hot_matrix, axis=0).reshape(1, -1)
