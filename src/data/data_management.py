import h5py
import numpy as np
import pickle
import scipy.io


class DataManagement:
    @staticmethod
    def load_dataset(path, train_set, test_set, vectorize=True, normalize=True):
        train_data = h5py.File(path + train_set, "r")
        train_x = np.array(train_data["train_set_x"][:])
        train_set_y_orig = np.array(train_data["train_set_y"][:])
        test_data = h5py.File(path + test_set, "r")
        test_x = np.array(test_data["test_set_x"][:])
        test_set_y_orig = np.array(test_data["test_set_y"][:])
        if vectorize:
            train_x = train_x.reshape(train_x.shape[0], -1).T
            test_x = test_x.reshape(test_x.shape[0], -1).T
            train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
            test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        if normalize:
            train_x = train_x / 255
            test_x = test_x / 255
        return train_x, train_set_y_orig, test_x, test_set_y_orig

    @staticmethod
    def load_dataset_2D_points(path):
        data = scipy.io.loadmat(path)
        train_x, train_y = data["X"].T, data["y"].T
        test_x, test_y = data["Xval"].T, data["yval"].T
        return train_x, train_y, test_x, test_y

    @staticmethod
    def save_model(path, name, model):
        with open(f"{path}/{name}.pickle", "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(path, name):
        with open(f"{path}/{name}.pickle", "rb") as f:
            model = pickle.load(f)
        return model



