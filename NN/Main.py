# Neural Network with one hidden layer

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def loadCSV (filePath, skip=1) :
    return np.asarray(pd.read_csv(filePath, skiprows=skip, header=None))


train = loadCSV('../train.csv')
test = loadCSV('../test.csv')

featureSize = 785
hiddenSize = 100
outputSize = 10

dataSize = 42000
testSize = 28000

reg = 5.0

maxIterations = 200
iteration = 1
cachedCost = 0

def sigmoid (x) :
    return 1.0 / (1 + np.exp(-x))

def sigmoidGrad (x) :
    return sigmoid(x) * (1 - sigmoid(x))

def predict (x, data, m, featureSize, hiddenSize, outputSize) :
    hiddenTheta = x[0 : featureSize * (hiddenSize - 1)].reshape((featureSize, (hiddenSize - 1)))
    outputTheta = x[featureSize * (hiddenSize - 1) : featureSize * (hiddenSize - 1) + hiddenSize * outputSize].reshape((hiddenSize, outputSize))
    
    # m x featureSize
    features = np.ones((m, featureSize))
    features[:, 1:] = data / 255.0

    # m x hiddenLayerSize
    hiddenLayer = np.ones((m, hiddenSize))
    hiddenLayer[:, 1:] = sigmoid(features.dot(hiddenTheta))
    
    # m x 
    outputLayer = sigmoid(hiddenLayer.dot(outputTheta))
    
    return np.argmax(outputLayer, 1)

def getCost (x, *args):
    data, featureSize, hiddenSize, outputSize, reg = args
    m = np.size(data, 0)
    
    hiddenTheta = x[0 : featureSize * (hiddenSize - 1)].reshape((featureSize, (hiddenSize - 1)))
    outputTheta = x[featureSize * (hiddenSize - 1) : featureSize * (hiddenSize - 1) + hiddenSize * outputSize].reshape((hiddenSize, outputSize))
    
    # m x 1
    labels = data[:, 0]
    expectedValues = np.eye(outputSize)[labels]
    
    # m x featureSize
    features = np.ones((m, featureSize))
    features[:, 1:] = data[:, 1:] / 255.0

    # m x hiddenSize
    hiddenLayer = np.ones((m, hiddenSize))
    hiddenLayer[:, 1:] = sigmoid(features.dot(hiddenTheta))
    
    # m x outputSize
    outputLayer = sigmoid(hiddenLayer.dot(outputTheta))
    
    J = -np.sum(expectedValues * np.log(outputLayer + 1e-8) + (1 - expectedValues) * np.log(1 - outputLayer + 1e-8)) / m
    J += reg / (2 * m) * (np.sum(hiddenTheta[:, 1:] ** 2) + np.sum(outputTheta[:, 1:] ** 2))
    
    # m x outputSize
    err3 = (outputLayer - expectedValues) * 1 # sigmoidGrad(hiddenLayer.dot(outputLayerTheta))
    
    # m x (hiddenSize - 1)
    err2 = err3.dot(outputTheta.transpose())[:, 1:] * sigmoidGrad(features.dot(hiddenTheta))
    
    # hiddenSize x outputSize
    grad2 = hiddenLayer.transpose().dot(err3) / m + reg * np.concatenate((np.zeros((hiddenSize, 1)), outputTheta[:, 1:]), axis = 1) / m
    
    # featureSize x (hiddenSize - 1)
    grad1 = features.transpose().dot(err2) / m + reg * np.concatenate((np.zeros((featureSize, 1)), hiddenTheta[:, 1:]), axis = 1) / m
    
    global cachedCost
    cachedCost = J
    return (J, np.concatenate((grad1.flatten(), grad2.flatten()), axis=0))

def getNumericalGradient (getCost, theta, args) :
    epsilon = 1e-5
    numgrad = np.zeros(np.size(theta))
    for i in range(np.size(theta, 0)):
        oldT = theta[i]
        theta[i] = oldT + epsilon
        pos = getCost(theta, *args)[0]
        theta[i] = oldT - epsilon
        neg = getCost(theta, *args)[0]
        numgrad[i] = (pos - neg) / (2 * epsilon)
        theta[i] = oldT
    return numgrad

def callback (xk) :
    global iteration
    print iteration
    print cachedCost
    iteration += 1

args = (train, featureSize, hiddenSize, outputSize, reg)

x0 = np.random.rand(featureSize * (hiddenSize - 1) + hiddenSize * outputSize) / 100.0
# print np.sum((getCost(x0, *args)[1] - getNumericalGradient(getCost, x0, args)) ** 2)
x1 = minimize(fun=getCost, x0=x0, method="CG", options={"maxiter": maxIterations, "disp":True}, jac=True, args=args, callback=callback).x

np.savetxt("../results.csv", predict(x1, test, testSize, featureSize, hiddenSize, outputSize), delimiter=",", fmt="\"%d\"")
np.savetxt("../theta.csv", x1, delimiter=",", fmt="%f")