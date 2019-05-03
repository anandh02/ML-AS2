from __future__ import division  # floating point division
import numpy as np
import math


import utilities as utils


class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.weights = None
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        self.weights = None
        try:
            utils.update_dictionary_items(self.params, parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def calc_error(self, Xtrain, ytrain, numsamples , error_list):
        error_list.append(np.sum( np.dot((np.dot(Xtrain, self.weights) - ytrain), np.dot(Xtrain, self.weights) - ytrain)) / numsamples)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest


class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.min = 0
        self.max = 1

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0]) * (self.max - self.min) + self.min
        return ytest


class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """

    def __init__(self, parameters={}):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.mean = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],)) * self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """

    def __init__(self, parameters={}):
        self.params = {'features': [1, 2, 3, 4, 5, 6]}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T, Xless) / numsamples), Xless.T), ytrain) / numsamples

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest




class RidgeLinearRegression(Regressor):

    def __init__(self, parameters={}):
        self.params = {'regwgt': 0.5}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):

        lamda = 0.01
        self.weights = np.dot(np.dot(np.linalg.inv(
            np.dot(Xtrain.T, Xtrain) + np.dot(lamda, (np.eye(Xtrain.shape[1], Xtrain.shape[1])))), Xtrain.T),
                              ytrain)

    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest


class gradientDescent(Regressor):

    BGDerror = []
    isBGDEnabled = False

    def __init__(self, parameters={}):
        # self.params = {'features': [1,2,3,4,5,6]}
        self.reset(parameters)
        self.epooch = 1000
        self.tolerance = 10e-4
        self.max_iteration = 10e5


    def learn(self, Xtrain, ytrain):

        numsamples = Xtrain.shape[0]
        self.weights = np.random.rand(Xtrain.shape[1])
        eta = 1
        iteration = 0
        isBGDEnabled = True
      #  self.BGDerror = []
        while True:

            c_w_o = np.dot((np.dot(Xtrain, self.weights) - ytrain), np.dot(Xtrain, self.weights) - ytrain) / (2 * numsamples)       # error calculate with W(t-1)
            grad = np.dot(Xtrain.T, np.dot(Xtrain, self.weights) - ytrain) / numsamples
            weights = self.weights - eta * grad
            c_w_n = np.dot((np.dot(Xtrain, weights) - ytrain), np.dot(Xtrain, weights) - ytrain) / ( 2 * numsamples)        # error calculated after gradient descent W

            """
            If the new calculated weights causing the error to increase then step size is reduced and this process is repeated until the error is reduced
            """

            while (c_w_o < c_w_n):
                eta = eta * 0.5
                weights = self.weights - eta * grad
                c_w_n = np.dot((np.dot(Xtrain, weights) - ytrain), np.dot(Xtrain, weights) - ytrain) / (  numsamples)
                iteration += 1
            self.weights = weights
            iteration += 1

            self.calc_error(Xtrain, ytrain, numsamples, self.BGDerror)

            if iteration > self.max_iteration or abs(c_w_n - c_w_o) < self.tolerance:
                print(str(iteration))
                break


    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest


class stochasticgradientDescent(Regressor):

    SGDerror = []
    SGDerrorEnabled = False

    def __init__(self, parameters={}):

        self.reset(parameters)
        self.epooch = 1000

    def learn(self, Xtrain, ytrain):
        eta = 0.01
        self.weights = np.random.rand(Xtrain.shape[1])
        self.SGDerrorEnabled = True
        #self.SGDerror = []
        iteration = 0
        for j in range(self.epooch):
            """
             shuffling the data for each epooch    
            """

            indices = np.arange(Xtrain.shape[0])
            np.random.shuffle(indices)
            Xtrain = Xtrain[indices]
            ytrain = ytrain[indices]

            for i in range(Xtrain.shape[0] - 1):
                gradient = np.dot(Xtrain[i].T, (np.dot(Xtrain[i].T, self.weights) - ytrain[i]))
                self.weights = self.weights - eta * gradient

            self.calc_error(Xtrain, ytrain, Xtrain.shape[0], self.SGDerror)
            iteration += 1
        print(iteration)

    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest


class LassoLinearRegression(Regressor):

    def __init__(self, parameters={}):
        self.params = {'regwgt': 0.5}
        self.reset(parameters)
        self.max_iteration = 10e3
        self.tolerance = 10e-4

    def learn(self, Xtrain, ytrain):

        numsamples = Xtrain.shape[0]
        self.weights = np.zeros(Xtrain.shape[1])

        iteration = 0

        eta = 1 / ( 2 * math.sqrt(np.sum(np.dot(Xtrain.T,Xtrain))))
        lamda = 0.01

        XX = np.dot(Xtrain.T,Xtrain) / numsamples
        Xy = np.dot(Xtrain.T,ytrain) / numsamples

        while True:

            c_w_o = np.dot((np.dot(Xtrain, self.weights) - ytrain), np.dot(Xtrain, self.weights) - ytrain) / (2 * numsamples)

            """  soft thresholding Calcuation """
            self.weights = self.weights - eta * (np.dot(XX, self.weights)) + eta * Xy
            self.soft_threshold(eta, lamda)

            c_w_n = np.dot((np.dot(Xtrain, self.weights) - ytrain), np.dot(Xtrain, self.weights) - ytrain) / (2 * numsamples)
            iteration += 1
            if iteration > self.max_iteration or abs(c_w_n - c_w_o) < self.tolerance:
                break

    def soft_threshold(self, eta, lamda):
        for i in range(self.weights.shape[0]):
            if self.weights[i] > eta * lamda:
                self.weights[i] -= eta * lamda
            elif (self.weights[i] < - (eta * lamda)):
                self.weights[i] += eta * lamda
            else:
                self.weights[i] = 0

    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest





class RMSPropRegression(Regressor):

    def __init__(self, parameters={}):
        # Default parameters, any of which can be overwritten by values passed to params
        #self.params = {'regwgt': 0.5}
        # self.params = {'features': range(30)}
        self.reset(parameters)
        self.epooch = 1000


    def learn(self, Xtrain, ytrain):

        eta = 0.01
        self.weights = np.random.rand(Xtrain.shape[1])
        E_G_t_1= 0
        for j in range(self.epooch):
            """
             shuffling the data for each epooch    
            """

            indices = np.arange(Xtrain.shape[0])
            np.random.shuffle(indices)
            Xtrain = Xtrain[indices]
            ytrain = ytrain[indices]


            for i in range(Xtrain.shape[0] - 1):
                gradient = np.dot(Xtrain[i].T, (np.dot(Xtrain[i].T, self.weights) - ytrain[i]))

                G_t_2 =  np.dot(gradient.T,gradient)

                Expected_value = 0.9 * E_G_t_1 + 0.1 * G_t_2
                self.weights = self.weights - eta * gradient / math.sqrt(Expected_value)

                E_G_t_1 = G_t_2 / (i + 1)


    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest


class amsGrad(Regressor):

    def __init__(self, parameters={}):
        self.reset(parameters)
        self.epooch = 1000

    def learn(self, Xtrain, ytrain):

        eta = 0.01
        self.weights = np.random.rand(Xtrain.shape[1])

        T = 1000
        v_t = 0
        m_t = 0
        beta1 = 0.9
        beta2 = 0.9
        v_cap_t = 0
        for j in range(self.epooch):
            """
             shuffling the data for each epooch    
            """
            indices = np.arange(Xtrain.shape[0])
            np.random.shuffle(indices)
            Xtrain = Xtrain[indices]
            ytrain = ytrain[indices]
            for i in range (T):

                """  Stroing the (t-1) value"""
                v_cap_t_1 = v_cap_t
                v_t_1 = v_t
                m_t_1 = m_t

                gradient = np.dot(Xtrain[i].T, (np.dot(Xtrain[i].T, self.weights) - ytrain[i]))
                m_t = beta1 * m_t_1 + (1 - beta1) * gradient
                v_t = beta2 * v_t_1 + (1 - beta2) * np.dot(gradient.T,gradient)
                v_cap_t = max( v_cap_t_1, v_t)
                self.weights = self.weights - eta* m_t / math.sqrt(v_cap_t)

    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest

