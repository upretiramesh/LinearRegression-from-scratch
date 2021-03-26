import numpy as np
import matplotlib.pyplot as plt

from ACIT4630.polynomialFeatures import PolynomialFeature
import ACIT4630.accuracyEvaluator as accuracy
from ACIT4630.regression import LinearRegression
import ACIT4630.preprocessing as ps

from sklearn.datasets import load_boston

print('########## PROCESS START ###############\n')
''' Load the features data '''
boston = load_boston()
data = boston.data


''' Load the target data '''
y = boston.target
y = np.expand_dims(y, axis=1)
print('Before removing outlier: Shape: ', data.shape, y.shape)


''' Remove null or fill null by other values'''
data, y = ps.drop_null(data, y)
data = ps.fill_null(data, method='bfill') # mean, median, ffill, bfill
print('\nRemove/fill null value completed')


'''Removing outlier from the dataset'''
data, y = ps.z_score_outlier_detection(data, y)
# data, y = ps.std_outlier_detection(data, y)
# data, y = ps.interquartile_range_outlier_detection(data, y)
print('\nAfter removing outlier: ', data.shape, y.shape)


''' Apply normalization to the data'''
data = ps.normalization(data)
# data = ps.standardscaler(data)
print('\nNormalization is completed')


'''Apply PolynomialFeature :default is degree 2'''
poly = PolynomialFeature() # default degree=2 and interaction_only=False
data = poly.transform(data=data)
print('\nConverted into PolynomialFeatures ')


''' Split data data into train and test set'''
train_size = 0.7
num = len(data)
X_train, X_rest = data[:int(num*train_size), :], data[int(num*train_size):, :]
y_train, y_rest = y[:int(num*train_size), :], y[int(num*train_size):, :]

train_size = 0.5
num = len(X_rest)
X_vall, X_test = X_rest[:int(num*train_size), :], X_rest[int(num*train_size):, :]
y_vall, y_test = y_rest[:int(num*train_size), :], y_rest[int(num*train_size):, :]
print('\nX_train:', X_train.shape, 'X_test: ', X_test.shape, 'X_vall: ', X_vall.shape)
print('\ny_train:', y_train.shape, 'y_test: ', y_test.shape, 'y_vall: ', y_vall.shape)


''' Define linear regression '''
model = LinearRegression()
model.fit(X_train, y_train, optim='GD', epochs=50, batch=25, lr=0.01) #, X_vall, y_vall, optim='MBGD', 'SGD'
print('\nModel training is complete')


''' Predict the response value'''
yhat = model.predict(X_test)
print('\nPrediction of response value is completed')


''' Model Evaluation'''
# print('\nMean square error: ', accuracy.mse_score(y_test, yhat))
print('\nRoot mean square error: ', accuracy.rmse_score(y_test, yhat))
# print('Residual sum of square:', accuracy.rss_score(y_test, yhat))
print('Mean absolute error: ', accuracy.mae_score(y_test, yhat))
# print('R-square error: ', accuracy.r2_score(y_test, yhat))


""" Plot the loss graph """
plt.figure(figsize=(10,6))
plt.plot(list(range(len(model.loss))), model.loss, label='Training')
if len(model.val_loss)!=0:
    plt.plot(list(range(len(model.val_loss))), model.val_loss, label='Validation')
plt.legend(loc='best')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()