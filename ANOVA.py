from base import Kernel
import numpy as np


class ANOVA(Kernel):
    def __init__(self, sigma=1., d=2):
        self._sigma = sigma
        self._d = d

    def _compute(self, data_1, data_2):
        kernel = np.zeros((data_1.shape[0], data_2.shape[0]))
        for d in range(data_1.shape[1]):
            column_1 = data_1[:, d].reshape(-1, 1)
            column_2 = data_2[:, d].reshape(-1, 1)
            kernel += np.exp(-self._sigma * (column_1 - column_2.T) ** 2) ** self._d
        return kernel

    def dim(self):
        return None
