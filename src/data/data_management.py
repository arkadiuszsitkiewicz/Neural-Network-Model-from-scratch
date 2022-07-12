import h5py
import numpy as np
import pickle


class DataManagement:
    @staticmethod
    def load_dataset(path, vectorize=True, normalize=True):
        # original shape - 209 examples of (64, 64, 3)
        train_data = h5py.File(path + "/train_catvnoncat.h5", "r")
        train_set_x_orig = np.array(train_data["train_set_x"][:])
        train_set_y_orig = np.array(train_data["train_set_y"][:])
        test_data = h5py.File(path + "/test_catvnoncat.h5", "r")
        test_set_x_orig = np.array(test_data["test_set_x"][:])
        test_set_y_orig = np.array(test_data["test_set_y"][:])
        if vectorize:
        # reshape to (64 * 64 * 3), 209
            train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
            test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
            train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
            test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        if normalize:
            train_x = train_set_x / 255
            test_x = test_set_x / 255
        return train_x, train_set_y_orig, test_x, test_set_y_orig

    @staticmethod
    def load_model(path, solver, learning_rate):
        with open(f"{path}/params_cost_{solver}_{learning_rate}.pickle", "rb") as f:
            model_data = pickle.load(f)
        return model_data["params"], model_data["cost"]

    @staticmethod
    def save_model(path, params, cost, solver, learning_rate):
        with open(f"{path}/params_cost_{solver}_{learning_rate}.pickle", "wb") as f:
            pickle.dump({"params": params, "cost": cost}, f)

