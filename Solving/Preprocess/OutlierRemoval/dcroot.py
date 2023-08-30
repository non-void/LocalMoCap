import numpy as np


def dcroot_plus(w, a, lbd, ubd):
    x = ubd
    a2 = a / 2
    l1 = lbd + a2 / np.sqrt(lbd)
    u1 = ubd + a2 / np.sqrt(ubd)
    I1 = np.where(w <= l1)
    if I1[0].shape[0]:
        x[I1] = lbd[I1]
    I2 = np.where(np.bitwise_and(w > l1, w < u1))
    if I2[0].shape[0]:
        v = w[I2] / 3
        x[I2] = 2 * v * (1 + np.cos((2 * np.pi / 3) - (2 / 3) * np.arccos(a2[I2] / (v ** 1.5) / 2)))
    return x


def dcroot_minus(w, a, lbd, ubd):
    x = lbd
    a2 = a / 2
    l1 = lbd - a2 / np.sqrt(lbd)
    u1 = ubd - a2 / np.sqrt(ubd)
    I1 = np.where(w >= u1)
    if I1[0].shape[0]:
        x[I1] = ubd[I1]
    I2 = np.where(np.bitwise_and(w > l1, w < u1))
    if I2[0].shape[0]:
        x[I2] = dcroot_minus_med(w[I2], a[I2])
    return x


def dcroot_minus_med(w, a):
    x = w
    a = a / 4
    w = w / 3
    a2 = a ** 2
    w3 = w ** 3
    d = a2 - w3
    I1 = np.where(d < 0)
    if I1[0].shape[0]:
        x[I1] = (2 * w[I1]) * (1 + np.cos((2 / 3) * np.arccos(np.sqrt(a2[I1] / w3[I1]))))
    I2 = np.where(np.bitwise_and(d >= 0, w >= 0))
    if I2[0].shape[0]:
        x[I2] = (a[I2] + np.sqrt(d[I2])) ** (1 / 3) + (a[I2] - np.sqrt(d[I2])) ** (1 / 3)
        x[I2] = x[I2] ** 2
    I3 = np.where(np.bitwise_and(d >= 0, w < 0))
    if I3[0].shape[0]:
        x[I3] = (np.sqrt(d[I3]) + a[I3]) ** (1 / 3) + (np.sqrt(d[I3]) - a[I3]) ** (1 / 3)
        x[I3] = x[I3] ** 2
    return x


def dcroot_plus_minus(w, a, d2, lbd, ubd):
    x = ubd
    a2 = a / 2
    l1 = lbd - a2 / np.sqrt(lbd)
    dl = d2 - a2 / np.sqrt(d2)
    du = d2 + a2 / np.sqrt(d2)
    u1 = ubd + a2 / np.sqrt(ubd)
    I1 = np.where(w <= l1)
    if I1[0].shape[0]:
        x[I1] = lbd[I1]
    I2 = np.where(np.bitwise_and(w > l1, w < dl))
    if I2[0].shape[0]:
        x[I2] = dcroot_minus_med(w[I2], a[I2])
    I3 = np.where(np.bitwise_and(w >= dl, w <= du))
    if I3[0].shape[0]:
        x[I3] = d2[I3]
    I4 = np.where(np.bitwise_and(w > du, w < u1))
    if I4[0].shape[0]:
        v = w[I4] / 3
        x[I4] = 2 * v * (1 + np.cos((2 * np.pi / 3) - (2 / 3) * np.arccos(a2[I4] / (v ** 1.5) / 2)))
    return x


def dcroot(w, a, Ind, d2, lbd, ubd):
    x = np.minimum(ubd, np.maximum(lbd, w))
    I1idx = np.where(d2[Ind] < lbd[Ind])
    I1 = Ind[0][I1idx], Ind[1][I1idx]
    if I1[0].shape[0]:
        x[I1] = dcroot_plus(w[I1], a[I1], lbd[I1], ubd[I1])
    I2idx = np.where(d2[Ind] >= ubd[Ind])
    I2 = Ind[0][I2idx], Ind[1][I2idx]
    if I2[0].shape[0]:
        x[I2] = dcroot_minus(w[I2], a[I2], lbd[I2], ubd[I2])
    I3idx = np.setdiff1d(np.arange(Ind[0].shape[0]), np.union1d(I1idx, I2idx))
    I3 = Ind[0][I3idx], Ind[1][I3idx]
    if I3[0].shape[0]:
        x[I3] = dcroot_plus_minus(w[I3], a[I3], d2[I3], lbd[I3], ubd[I3])
    return x.real
