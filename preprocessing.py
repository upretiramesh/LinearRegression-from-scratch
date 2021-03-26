import numpy as np

"""
Detect the null value any fix it

Methods:
1. Remove the null value
2. Fill the null value by mean
3. Fill the mean value by median

"""


def drop_null(data, y):
    for i in range(data.shape[1]):
        idx = np.where(np.isnan(data[:, i]) == True)
        if len(idx[0]) != 0:
            data = np.delete(data, idx[0], axis=0)
            y = np.delete(y, idx[0], axis=0)
    return data, y


def fill_null(data, method='mean'):
    if method == 'mean':
        for i in range(data.shape[1]):
            idx = np.where(np.isnan(data[:, i]) == True)
            if len(idx[0]) != 0:
                data[idx[0], i] = np.nanmean(data[:, i])
        return data
    elif method == 'median':
        for i in range(data.shape[1]):
            idx = np.where(np.isnan(data[:, i]) == True)
            if len(idx[0]) != 0:
                data[idx[0], i] = np.nanmedian(data[:, i])
        return data
    elif method == 'ffill':
        for i in range(data.shape[1]):
            idx = np.where(np.isnan(data[:, i]) == True)
            if len(idx[0]) != 0:
                for index in idx[0]:
                    data[index, i] = data[index - 1, i]
        return data
    elif method == 'bfill':
        for i in range(data.shape[1]):
            idx = np.where(np.isnan(data[:, i]) == True)
            if len(idx[0]) != 0:
                for index in idx[0][-1::]:
                    data[index, i] = data[index + 1, i]
        return data
    else:
        print('Invalid choice please select :: mean, median, ffill, bfill')


"""
Remove outlier 

Methods:

1. Standard deviation method
2. Z-scores
3. IQR         
"""


def std_outlier_detection(data, y):
    lower = np.mean(data, axis=0) - np.std(data, axis=0)*3
    upper = np.mean(data, axis=0) + np.std(data, axis=0)*3
    idx = []
    # print('lower: ', lower, 'upper: ', upper)
    for i in range(data.shape[1]):
        indexs = np.where((data[:, i] < lower[i])|(data[:, i] > upper[i]))
        if list(indexs[0]):
            idx.extend(list(indexs[0]))
    if idx:
        print('std_outlier_detection: ', len(set(idx)))
        data = np.delete(data, list(set(idx)), axis=0)
        y = np.delete(y, list(set(idx)), axis=0)

    return data, y


def z_score_outlier_detection(data, y):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    idx = []
    for i in range(data.shape[1]):
        if std[i]!=0:
            z = (data[:, i]-mean[i])/std[i]
            indexs = np.where((z>3) | (z<-3))
            if list(indexs[0]):
                idx.extend(list(indexs[0]))
    if idx:
        print('z-score outlier detection removed: ', len(set(idx)))
        data = np.delete(data, list(set(idx)), axis=0)
        y = np.delete(y, list(set(idx)), axis=0)

    return data, y


def interquartile_range_outlier_detection(data, y):
    Q1 = np.quantile(data, 0.25, axis=0)
    Q3 = np.quantile(data, 0.75, axis=0)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    idx = []
    for i in range(data.shape[1]):
        indexs = np.where((data[:, i] < lower[i]) | (data[:, i] > upper[i]))
        if list(indexs[0]):
            idx.extend(list(indexs[0]))
    if idx:
        print('Interquartile range outlier detection removed: ', len(set(idx)))
        data = np.delete(data, list(set(idx)), axis=0)
        y = np.delete(y, list(set(idx)), axis=0)

    return data, y




"""
Data normalization

Methods:
1. Normalizer 
2. StandardScaler
3. MinMaxScaler
4. MaxAbsScaler

"""


def normalization(data):
    col_min = np.min(data, axis=0)
    col_max = np.max(data, axis=0)
    for i in range(data.shape[1]):
        if col_min[i]!=col_max[i]:
            data[:, i] = (data[:, i] - col_min[i]) / (col_max[i] - col_min[i])
    return data


def standardscaler(data):
    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    for i in range(data.shape[1]):
        data[:, i] = (data[:, i] - mean[i]) / sd[i]
    return data

