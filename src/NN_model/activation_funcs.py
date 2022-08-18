import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z, out=np.full_like(Z, np.inf), where=(-Z <= 709.78)))
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
    s = 1 / (1 + np.exp(-Z, out=np.full_like(Z, np.inf), where=(-Z <= 709.78)))
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


def softmax(Z):
    exps = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = exps / np.sum(exps, axis=0, keepdims=True)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache


def softmax_backward(dA, cache, mask):
    Z = cache
    r, c = Z.shape
    exps = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    s = (exps / np.sum(exps, axis=0, keepdims=True)).reshape(r * c, 1, order="F")
    ds = s * np.identity(s.size) - mask * np.dot(s, s.T)
    dZ = np.dot(ds, dA.reshape(r * c, 1, order="F")).reshape(r, c, order="F")
    assert (dZ.shape == Z.shape)
    return dZ


