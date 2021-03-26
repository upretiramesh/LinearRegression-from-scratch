import numpy as np


class Normalization:
    def __init__(self, method='nm', ignore_null=True):
        self.method = method

    def apply(self, nor_data):
        if self.method == 'nm':
            for i in range(nor_data.shape[1]):
                col_min = np.min(nor_data[:, i])
                col_max = np.max(nor_data[:, i])
                nor_data[:, i] = (nor_data[:, i] - col_min) / (col_max - col_min)
            return nor_data

        elif self.method == 'sds':
            for i in range(nor_data.shape[1]):
                mean = np.mean(nor_data[:, i])
                sd = np.std(nor_data[:, i])
                nor_data[:, i] = (nor_data[:, i] - mean) / sd
            return nor_data
        else:
            print('Wrong argument value: select the right normalization method')
