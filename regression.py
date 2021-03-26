import numpy as np


class LinearRegression:
    def __init__(self):
        self.weights = 0
        self.bias = 0
        self.optim = 'GD'
        self.batch = 25
        self.epochs = 20
        self.lr = 0.01
        self.loss = []
        self.val_loss = []

    def __train(self, Xtrain, Xtest):
        """
        :param Xtrain: train data
        :param Xtest: target value
        :return: None
        """

        # predict the target value
        yhat = Xtrain @ self.weights + self.bias

        # calculate error
        error = Xtest - yhat
        if self.optim == 'GD':
            self.loss.append(np.mean(error))

        num = len(Xtrain)
        # calculate the derivative of weights and bias
        derivative_weights = (-2 / num) * (Xtrain.transpose() @ error)
        derivative_bias = (-2 / num) * np.sum(error)

        # update weights and bias
        self.weights = self.weights - self.lr * derivative_weights
        self.bias = self.bias - self.lr * derivative_bias

    def __validation_loss(self, Vtrain, Vtest):
        yhat = Vtrain @ self.weights + self.bias
        error = Vtest - yhat
        self.val_loss.append(np.mean(error))

    def fit(self, X, y, Vx=None, Vy=None, optim=None, batch=None, epochs=None, lr=None):
        """
        :param X: data for training
        :param y: target value
        :param optim: select optimizer/cost function [default: Gradient Descent-GD,
                Mini-Batch Gradient Descent - MBGD, Stochastic Gradient Descent - SGD
        :param batch: default batch=25, enter value if you want to change
        :param epochs: default epochs=1000, enter if value want to change
        :param lr: default lr=0.001, enter new learning rate if you want to change
        :return: updates the weights and bias
        """

        if optim is not None:
            self.optim = optim
        if batch is not None:
            self.batch = batch
        if epochs is not None:
            self.epochs = epochs
        if lr is not None:
            self.lr = lr

        # Initialize the weights based on the training features
        self.weights = np.zeros((X.shape[1], 1))

        # Calculate the total length of training dataset

        if self.optim == 'GD':
            for i in range(self.epochs):
                self.__train(X, y)

            # if Vx is not None and Vy is not None:
            #     self.__validation_loss(Vx, Vy)

        elif self.optim == 'MBGD':
            # calculate number of batches
            batches = int(len(X) / self.batch)

            print('Mini Batch Gradient Decent is Running')
            for epoch in range(self.epochs):

                for i in range(batches + 1):
                    if i == batches:
                        if len(X[i * self.batch:, :]) != 0:
                            self.__train(X[i * self.batch:, :], y[i * self.batch:, :])
                    else:
                        self.__train(X[i * self.batch:i * self.batch + self.batch, :],
                                     y[i * self.batch:i * self.batch + self.batch, :])

                if Vx is not None and Vy is not None:
                    self.val_loss.append(self.__validation_loss(Vx, Vy))

        elif self.optim == 'SGD':
            for epoch in range(self.epochs):
                indexes = np.arange(len(X))
                np.random.shuffle(indexes)
                for idx in indexes:
                    self.__train(X[idx:idx + 1, :], y[idx:idx + 1, :])
                if Vx is not None and Vy is not None:
                    self.val_loss.append(self.__validation_loss(Vx, Vy))
        else:
            print('Wrong Argument Value : Select the right optimizer from (GD, MBGD, SGD)')

    def predict(self, test):
        """

        :param test: test data
        :return: predicted value of target
        """

        yhat = test @ self.weights + self.bias

        return yhat
