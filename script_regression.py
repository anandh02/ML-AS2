from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt


import dataloader as dtl
import regressionalgorithms as algs

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]


if __name__ == '__main__':
    trainsize = 1000
    testsize = 5000
    numruns = 1

    """
    By changing the value in feature variable it will change the number of features to be selected
    """
    L = []
    feature = 5
    L.extend(range(feature))


    regressionalgs = {#'Random': algs.Regressor(),
                'Mean': algs.MeanPredictor(),
                'FSLinearRegression5': algs.FSLinearRegression({'features': L}),
                'FSLinearRegression50': algs.FSLinearRegression({'features': range(50)}),
                'RidgeLinearRegression': algs.RidgeLinearRegression(),
                'gradientDescent' : algs.gradientDescent(),
                'stochasticgradientDescent':algs.stochasticgradientDescent(),
                'LassoLinearRegression' : algs.LassoLinearRegression(),
                'RMSPropRegression' : algs.RMSPropRegression(),
                'amsGrad':algs.amsGrad(),
              }
    numalgs = len(regressionalgs)

    # Enable the best parameter to be selected, to enable comparison
    # between algorithms with their best parameter settings
    parameters = (
        #{'regwgt': 0.0},
        {'regwgt': 0.01},
        #{'regwgt': 1.0},
                      )
    numparams = len(parameters)
    
    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_ctscan(trainsize,testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in regressionalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])

                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)


                print ('Error for ' + learnername + ': ' + str(error))
                #print ('Standard Error' +learnername + ': ' + str(standard_error))
                errors[learnername][p,r] = error



    for learnername in regressionalgs:
        besterror = np.mean(errors[learnername][0,:])
        standard_error = np.std(errors[learnername]) / math.sqrt(numparams)
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p
        # Extract best parameters
        learner.reset(parameters[bestparams])
        #print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror))
        #print('Standard error for ' + learnername + ': ' + str(standard_error))

    print(algs.gradientDescent.BGDerror)
    print(algs.stochasticgradientDescent.SGDerror)
    l= []
    #algs.stochasticgradientDescent.SGDerror
    l.extend(range(len(algs.gradientDescent.BGDerror)))
    plt.plot(l, algs.gradientDescent.BGDerror,  lw = 0.9, color = 'k')
    print(algs.gradientDescent.isBGDEnabled)

    m = []
    m.extend(range(len(algs.stochasticgradientDescent.SGDerror)))
    plt.plot(l ,algs.stochasticgradientDescent.SGDerror, lw = 0.9, color = 'r')
    plt.show()