import numpy as np


def mse_score(real, predict):
    return np.sum((real - predict) ** 2) / len(real)


def rmse_score(real, predict):
    return np.sqrt(np.sum((real - predict) ** 2) / len(real))


def rss_score(real, predict):
    return np.sum((real - predict) ** 2)


def mae_score(real, predict):
    return np.sum(np.abs(real - predict)) / len(real)


def r2_score(real, predict):
    SSR = np.sum((real - predict) ** 2)
    SST = np.sum((real - np.mean(real)) ** 2)
    return 1 - SSR / SST
