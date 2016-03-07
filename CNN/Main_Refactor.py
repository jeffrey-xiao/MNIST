# Convolutional Neural Network:
# Input -> [Conv -> Relu -> Pool] -> FC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.optimize import minimize

np.random.seed(1)

def loadCSV (filePath, skip=1, dtype=np.float) :
    return np.asarray(pd.read_csv(filePath, skiprows=skip, header=None, dtype={'a': np.float}))

train = np.random.permutation(loadCSV('../train.csv', skip=1, dtype=np.float))
test = loadCSV('../test.csv', skip=1, dtype=np.float)

inputSize = 28

filterSize = 9
numFilters = 10 #10;32

poolSize = 2 # 2
pooledDim = (inputSize - filterSize + 1) / poolSize

outputSize = 10

reg = 0.003
maxIterations = 2000
iteration = 0
costCache = 0
accuracyCache = 0

def sigmoid (x) :
    return 1.0 / (1 + np.exp(-x) + 1e-8)

def sigmoidGrad (x) :
    return sigmoid(x) * (1 - sigmoid(x))

def unpackTheta (theta, filterSize, numFilters, pooledDim, outputSize):
    a = filterSize ** 2 * numFilters
    b = a + numFilters
    c = b + pooledDim ** 2 * numFilters * outputSize
    d = c + outputSize
    return (np.reshape(theta[0 : a], (filterSize, filterSize, numFilters)),
            np.reshape(theta[a : b], (numFilters)),
            np.reshape(theta[b : c], (pooledDim ** 2 * numFilters, outputSize)),
            np.reshape(theta[c : d], (outputSize)))

def predict (theta, *args) :
    ## Initializing variables
    features, inputSize, filterSize, numFilters, poolSize, outputSize = args
    filteredSize = inputSize - filterSize + 1
    pooledSize = filteredSize / poolSize
    (Wc, Bc, Wr, Br) = unpackTheta(theta, filterSize, numFilters, pooledSize, outputSize)
    numImages = np.size(features, 0)
    
    features = np.reshape(features / 255.0, (inputSize, inputSize, numImages))
    
    ## Forward propagation
    # Convolution
    conv = getConvolution(features, Wc, Bc, numFilters, numImages, filteredSize)
    
    # Mean pooling
    pooled = np.reshape(getPooled(conv, poolSize, pooledSize, filteredSize, numFilters, numImages), (-1, numImages))
    
    # Logistic regression
    calcValues = sigmoid(pooled.transpose().dot(Wr) + Br)
    
    return np.argmax(calcValues, 1)

def getConvolution (features, Wc, Bc, numFilters, numImages, filteredSize):
    conv = np.zeros((filteredSize, filteredSize, numFilters, numImages))
    for imageNum in range(numImages):
        image = features[:, :, imageNum]
        for filterNum in range(numFilters):
            filterLayer = np.rot90(Wc[:, :, filterNum], 2)
            conv[:, :, filterNum, imageNum] = sigmoid(convolve2d(image, filterLayer, 'valid') + Bc[filterNum])
    return conv

def getPooled (features, poolSize, pooledSize, filteredSize, numFilters, numImages):
    ret = np.empty((pooledSize, pooledSize, numFilters, numImages))
    poolFilter = np.ones((poolSize, poolSize)) / poolSize ** 2
    for i in range(numFilters): 
        for j in range(numImages):
            pooledLayer = convolve2d(features[:, :, i, j], poolFilter, 'valid')
            for x in range(0, filteredSize, poolSize):
                for y in range(0, filteredSize, poolSize):
                    ret[x / poolSize, y / poolSize, i, j] = pooledLayer[x, y]
    return ret

def getCost (theta, *args) :
    ## Initializing variables
    global accuracyCache
    features, inputSize, filterSize, numFilters, poolSize, outputSize, reg = args
    filteredSize = inputSize - filterSize + 1
    pooledSize = filteredSize / poolSize
    (Wc, Bc, Wr, Br) = unpackTheta(theta, filterSize, numFilters, pooledSize, outputSize)
    numImages = np.size(features, 0)
    
    labels = features[:, 0]
    actualValues = np.eye(outputSize)[labels]
    features = np.reshape(features[:, 1:] / 255.0, (inputSize, inputSize, numImages))
    
    ## Forward propagation
    # Convolution
    conv = getConvolution(features, Wc, Bc, numFilters, numImages, filteredSize)
    
    # Mean pooling
    pooled = np.reshape(getPooled(conv, poolSize, pooledSize, filteredSize, numFilters, numImages), (-1, numImages))
    
    # Logistic regression
    calcValues = sigmoid(pooled.transpose().dot(Wr) + Br)
    
    ## Calculating cost
    cost = -np.sum(actualValues * np.log(calcValues) + (1 - actualValues) * np.log(1 - calcValues)) / numImages
    cost += reg * (np.sum(Wc ** 2) + np.sum(Wr ** 2)) / 2
    
    accuracyCache = np.sum(np.argmax(calcValues, 1) == labels) / 1.0 / numImages
    
    ## Back propagation

    outputError = (calcValues - actualValues) / numImages
    
    pooledError = Wr.dot(outputError.transpose())
    pooledError = np.reshape(pooledError, (pooledSize, pooledSize, numFilters, numImages))
    
    convError = np.empty((filteredSize, filteredSize, numFilters, numImages))
    convErrorFilter = np.ones((poolSize, poolSize)) / (poolSize ** 2)
    for i in range(numFilters):
        for j in range(numImages):
            convError[:, :, i, j] = np.kron(pooledError[:, :, i, j], convErrorFilter)
    
    convError *= conv * (1 - conv)
    
    ## Gradient
    
    WrGrad = pooled.dot(outputError)
    BrGrad = np.sum(outputError, 0)
    
    WcGrad = np.zeros((filterSize, filterSize, numFilters))
    BcGrad = np.empty(numFilters)
    
    for i in range(numFilters):
        BcGrad[i] = np.sum(convError[:, :, i, :])
        for j in range(numImages): 
            filterLayer = np.rot90(convError[:, :, i, j], 2)
            WcGrad[:, :, i] += convolve2d(features[:, :, j], filterLayer, 'valid')
    
    WrGrad += reg * Wr
    WcGrad += reg * Wc
    
    global costCache
    costCache = cost
    return (cost, np.concatenate((WcGrad.flatten(), BcGrad.flatten(), WrGrad.flatten(), BrGrad.flatten()), axis=0))

def getNumericalGradient (getCost, theta, args, grad) :
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
        print (numgrad[i] - grad[i]) ** 2
        if i % 100 == 0:
            print i
    return numgrad

def SGD (costFunction, theta, options, args) :
    data, inputDim, filterDim, numFilters, poolDim, outputSize, reg = args
    
    epochs = options['epochs']
    alpha = options['alpha']
    minibatch = options['minibatch']
    momentum = options['momentum']
    
    m = np.size(args[0], 0)
    
    mom = 0.5;
    momIncrease = 20;
    velocity = 0
    it = 0
    
    res = []
    
    for i in range(epochs):
        for j in range(0, m - minibatch + 1, minibatch):
            it += 1

            if iteration == momIncrease:
                mom = momentum;

            (cost, grad) = costFunction(theta, data[j : j + minibatch, :], inputDim, filterDim, numFilters, poolDim, outputSize, reg)
            # (cost, grad) = costFunction(theta, data[0 : 100, :], inputDim, filterDim, numFilters, poolDim, outputSize, reg)
            
            # theta -= alpha * grad
            
            # grad2 = getNumericalGradient(getCost, theta, (data[0 : 100, :], inputDim, filterDim, numFilters, poolDim, outputSize, reg))
            # print np.sum((grad - grad2) ** 2)
            
            velocity = mom * velocity + alpha * grad;
            theta = theta - velocity
            
            print "Iteration: %d; Accuracy: %.2f; Cost: %.5f" %(it, accuracyCache, cost)
            res += [cost]
            if it % 1000 == 0:
                plt.plot(res[10:])
                plt.show()
                
        alpha /= 2.0
    return theta


def callback (xk) :
    global iteration
    global costCache
    global accuracyCache
    print "Iteration: %d; Accuracy: %.2f; Cost: %.5f" %(iteration, accuracyCache, costCache)
    print predict(xk, train[0 : 100, 1:], inputSize, filterSize, numFilters, poolSize, outputSize)
    iteration += 1
    
args = (train[100 : 1000], inputSize, filterSize, numFilters, poolSize, outputSize, reg)
debug = False

x0 = np.random.rand(filterSize ** 2 * numFilters + numFilters + pooledDim ** 2 * numFilters * outputSize + outputSize) / 100
print np.size(x0)

if debug:
    grad1 = getCost(x0, *args)[1]
    print grad1
    grad2 = getNumericalGradient(getCost, x0, args, grad1)
    print grad2
    
    print np.sum((grad1 - grad2) ** 2)
    assert (np.sum((grad1 - grad2) ** 2) < 1e-8)
else :
    x1 = minimize(fun=getCost, x0=x0, method="CG", options={"maxiter": maxIterations, "disp":True}, jac=True, args=args, callback=callback).x
    # x1 = SGD(getCost, x0, options={'epochs':1000, 'alpha': 0.003, 'minibatch': 100, 'momentum':0.95}, args=args)
    print predict(x1, train[0 : 100, 1 :], inputSize, filterSize, numFilters, poolSize, outputSize)
    np.savetxt("../resultsCNN.csv", predict(x1, test[0 : 100], inputSize, filterSize, numFilters, poolSize, outputSize), delimiter=",", fmt="\"%d\"")
