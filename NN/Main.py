# Neural Network with one hidden layer

import numpy as np
from scipy.optimize import minimize
import csv

def loadCSV (filePath, datatype = np.int) :
    with open(filePath,'r') as dest_file:
        data_iter = csv.reader(dest_file, delimiter=",", quotechar='"')
        data = [line for line in data_iter][1:]
    return np.asarray(data, dtype=datatype)


train = loadCSV('../train.csv')
test = loadCSV('../test.csv')

featureSize = 785
hiddenSize = 100
outputSize = 10

dataSize = 42000
testSize = 28000

reg = 5.0

maxIterations = 10
iteration = 1

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
    
    # 42000 x 10
    outputLayer = sigmoid(hiddenLayer.dot(outputTheta))
    
    J = -np.sum(expectedValues * np.log(outputLayer + 0.000001) + (1 - expectedValues) * np.log(1 - outputLayer + 0.000001)) / m
    J += reg / (2 * m) * (np.sum(hiddenTheta[:, 1:] ** 2) + np.sum(outputTheta[:, 1:] ** 2))
    
    # m x outputSize
    err3 = (outputLayer - expectedValues) * 1 # sigmoidGrad(hiddenLayer.dot(outputLayerTheta))
    
    # m x (hiddenSize - 1)
    err2 = err3.dot(outputTheta.transpose())[:, 1:] * sigmoidGrad(features.dot(hiddenTheta))
    
    hiddenTheta[:, 0] = 0
    outputTheta[:, 0] = 0
    
    # hiddenSize x outputSize
    grad2 = hiddenLayer.transpose().dot(err3) / m + reg * outputTheta / m
    
    # featureSize x (hiddenSize - 1)
    grad1 = features.transpose().dot(err2) / m + reg * hiddenTheta / m
    
    return (J, np.concatenate((grad1.flatten(), grad2.flatten()), axis=0))

def callback (xk) :
    global iteration
    print iteration
    iteration += 1

print "Before learning"

x0 = np.random.rand(featureSize * (hiddenSize - 1) + hiddenSize * outputSize) / 100.0

np.savetxt("../results.csv", predict(x0, test, testSize, featureSize, hiddenSize, outputSize), delimiter=",", fmt="\"%d\"")

args = (train, featureSize, hiddenSize, outputSize, reg)

x1 = minimize(fun=getCost, x0=x0, method="CG", options={"maxiter": maxIterations, "disp":True}, jac=True, args=args, callback=callback).x

np.savetxt("../results.csv", predict(x1, test, testSize, featureSize, hiddenSize, outputSize), delimiter=",", fmt="\"%d\"")
np.savetxt("../theta.csv", x1, delimiter=",", fmt="%f")