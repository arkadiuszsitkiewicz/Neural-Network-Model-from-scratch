import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache


def tanh(Z):
    A = np.tanh(Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 -s)
    assert (dZ.shape == Z.shape)
    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ


def tanh_backward(dA, cache):
    Z = cache
    s = np.tanh(Z)
    dZ = dA * (1 - s ** 2)
    assert (dZ.shape == Z.shape)
    return dZ
