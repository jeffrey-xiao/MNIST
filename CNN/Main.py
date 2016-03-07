# Convolutional Neural Network:
# Input -> [Conv -> Relu -> Pool] -> FC

import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from scipy.optimize import minimize

def loadCSV (filePath, skip=1) :
    return np.asarray(pd.read_csv(filePath, skiprows=skip, header=None))

np.random.seed(1)

train = loadCSV('../train.csv')
test = loadCSV('../test.csv')

inputDim = 28

filterDim = 9 # 9
numFilters = 1 # 32

poolDim = 20 # 2

assert((inputDim - filterDim + 1) % poolDim == 0)
pooledDim = (inputDim - filterDim + 1) / poolDim

outputSize = 10

reg = 0.0

maxIterations = 1000
costCache = 0
iteration = 1

def sigmoid (x) :
    return 1.0 / (1 + np.exp(-x) + 1e-6)

def sigmoidGrad (x) :
    return sigmoid(x) * (1 - sigmoid(x))

def unpack (x, filterDim, numFilters, pooledDim, outputSize):
    a = filterDim * filterDim * numFilters
    b = a + numFilters
    c = b + pooledDim * pooledDim * numFilters * outputSize
    d = c + outputSize
    Wc = x[0 : a] # Convolutional Layer Theta
    Bc = x[a : b] # Convolutional Layer Bias
    Wr = x[b : c] # Fully Connected Layer Theta
    Br = x[c : d] # Fully Connected Layer Bias
    return (np.reshape(Wc, (filterDim, filterDim, numFilters)),
            np.reshape(Bc, (numFilters)),
            np.reshape(Wr, (pooledDim * pooledDim * numFilters, outputSize)),
            np.reshape(Br, (outputSize)))

def predict (x, *args) :
    data, inputDim, filterDim, numFilters, poolDim, pooledDim, outputSize = args
    filteredDim = (inputDim - filterDim + 1)
    features = np.copy(data)
    numImages = np.size(features, 0)
    
    (Wc, Bc, Wr, Br) = unpack(x, filterDim, numFilters, pooledDim, outputSize)
    
    data = features / 255.0
    data = np.reshape(data, (inputDim, inputDim, numImages), 'C')
    
    ## Forward feed
    
    # Convolution Layer
    conv = np.zeros((filteredDim, filteredDim, numFilters, numImages))
    
    for imageNum in range(numImages):
        image = data[:, :, imageNum]
        for filterNum in range(numFilters):
            filterLayer = np.rot90(Wc[:, :, filterNum], 2)
            conv[:, :, filterNum, imageNum] = sigmoid(convolve2d(image, filterLayer, 'valid') + Bc[filterNum])

    # Pooling Layer
    
    pool = np.zeros((pooledDim, pooledDim, numFilters, numImages))
    poolFilter = np.ones((poolDim, poolDim)) / (poolDim ** 2)
    
    for imageNum in range(numImages):
        for filterNum in range(numFilters):
            convLayer = conv[:, :, filterNum, imageNum]
            pooledLayer = convolve2d(convLayer, poolFilter, 'valid')
            for x in range(0, filteredDim, poolDim):
                for y in range(0, filteredDim, poolDim):
                    pool[x / poolDim, y / poolDim, filterNum, imageNum] = pooledLayer[x, y]
    
    pool = np.reshape(pool, (-1, numImages))
    
    # Output Layer
    calc = sigmoid(pool.transpose().dot(Wr) + Br)
    print calc
    return np.argmax(calc, 1)

def getCost (x, *args) :
    data, inputDim, filterDim, numFilters, poolDim, pooledDim, outputSize = args
    filteredDim = (inputDim - filterDim + 1)
    features = np.copy(data)
    numImages = np.size(features, 0)
    
    (Wc, Bc, Wr, Br) = unpack(x, filterDim, numFilters, pooledDim, outputSize)
    
    labels = features[:, 0]
    actual = np.eye(outputSize)[labels]
    
    features = features[:, 1:] / 255.0
    features = np.reshape(features, (inputDim, inputDim, numImages), 'C')
    
    ## Forward feed
    
    # Convolution Layer
    conv = np.zeros((filteredDim, filteredDim, numFilters, numImages))
    for imageNum in range(numImages):
        image = features[:, :, imageNum]
        for filterNum in range(numFilters):
            filterLayer = np.rot90(Wc[:, :, filterNum], 2)
            conv[:, :, filterNum, imageNum] = sigmoid(convolve2d(image, filterLayer, 'valid') + Bc[filterNum])
    # Pooling Layer
    
    pool = np.zeros((pooledDim, pooledDim, numFilters, numImages))
    poolFilter = np.ones((poolDim, poolDim)) / (poolDim ** 2)
    
    for imageNum in range(numImages):
        for filterNum in range(numFilters):
            convLayer = conv[:, :, filterNum, imageNum]
            pooledLayer = convolve2d(convLayer, poolFilter, 'valid')
            for x in range(0, filteredDim, poolDim):
                for y in range(0, filteredDim, poolDim):
                    pool[x / poolDim, y / poolDim, filterNum, imageNum] = pooledLayer[x, y]
    
    pool = np.reshape(pool, (-1, numImages))
    # Output Layer
    calc = sigmoid(pool.transpose().dot(Wr) + Br)

    ## Calculating cost
    cost = -np.sum(actual * np.log(calc + 1e-8) + (1 - actual) * np.log(1 - calc + 1e-8)) / numImages
    cost += reg * (np.sum(Wc ** 2) + np.sum(Wr ** 2)) / (2 * numImages)
    ## Back propagation
    
    outputError = (calc - actual) / numImages
    
    pooledError = Wr.dot(outputError.transpose())
    pooledError = np.reshape(pooledError, (pooledDim, pooledDim, numFilters, numImages))
    
    poolingError = np.zeros((filteredDim, filteredDim, numFilters, numImages))
    for imageNum in range(numImages):
        for filterNum in range(numFilters):
            poolingError[:, :, filterNum, imageNum] = np.kron(pooledError[:, :, filterNum, imageNum], poolFilter)
    
    convError = poolingError * (conv) * (1 - conv) # derivative of sigmoid = sigmoid  * (1 - sigmoid)
    
    ## Gradient
    WrGrad = pool.dot(outputError)
    BrGrad = np.sum(outputError, 0)
    
    WcGrad = np.zeros((filterDim, filterDim, numFilters))
    BcGrad = np.zeros(numFilters)

    for imageNum in range(numImages):
        for filterNum in range(numFilters):
            filterLayer = np.rot90(convError[:, :, filterNum, imageNum], 2)
            WcGrad[:, :, filterNum] += convolve2d(features[:, :, imageNum], filterLayer, 'valid')
    
    for filterNum in range(numFilters):
        BcGrad[filterNum] = np.sum(convError[:, :, filterNum, :])
    
    WcGrad += reg * Wc / numImages
    WrGrad += reg * Wr / numImages
    
    global costCache
    costCache = cost
    return (cost, np.concatenate((WcGrad.flatten(), BcGrad.flatten(), WrGrad.flatten(), BrGrad.flatten()), axis=0))

def getNumericalGradient (getCost, theta, args) :
    epsilon = 1e-8
    numgrad = np.zeros(np.size(theta))
    for i in range(np.size(theta, 0)):
        oldT = theta[i]
        theta[i] = oldT + epsilon
        pos = getCost(theta, *args)[0]
        theta[i] = oldT - epsilon
        neg = getCost(theta, *args)[0]
        numgrad[i] = (pos - neg) / (2 * epsilon)
        theta[i] = oldT
        if i % 100 == 0:
            print i
    return numgrad

def callback (xk) :
    global iteration
    print iteration
    print costCache
    #print xk
    iteration += 1

def SGD (costFunction, theta, options, args) :
    data, inputDim, filterDim, numFilters, poolDim, pooledDim, outputSize = args
    epochs = options['epochs']
    alpha = options['alpha']
    minibatch = options['minibatch']
    
    mom = 0.5
    momIncrease = 20
    m = np.size(args[0], 0)
    
    velo = np.zeros(np.size(theta))
    
    it = 0
    for i in range(epochs):
        for j in range(0, m - minibatch + 1, minibatch):
            it += 1
            if it == momIncrease:
                mom = 0.9
                
            (cost, grad) = costFunction(theta, data[j : j + minibatch, :], inputDim, filterDim, numFilters, poolDim, pooledDim, outputSize)

            velo = mom * velo + alpha * grad
            theta -= alpha * grad
            
            print str(it) + " " + str(cost)
            print j
        alpha /= 2.0
    return theta

args = (train[0 : 100], inputDim, filterDim, numFilters, poolDim, pooledDim, outputSize)

x0 = np.random.rand(filterDim ** 2 * numFilters + numFilters + pooledDim ** 2 * numFilters * outputSize + outputSize) / 100
print np.size(x0)

grad1 = getCost(x0, *args)[1]
print grad1
grad2 = getNumericalGradient(getCost, x0, args)
print grad2
print np.sum((grad1 - grad2) ** 2)
assert (np.sum((grad1 - grad2) ** 2) < 1e-8)

'''
# x1 = minimize(fun=getCost, x0=x0, method="CG", options={"maxiter": maxIterations, "disp":True}, jac=True, args=args, callback=callback).x
#x1 = SGD(getCost, x0, options={'epochs':3, 'alpha': 0.1, 'minibatch': 10}, args=args)
print "Printing Theta"
print x0
print x1
np.savetxt("../resultsCNN.csv", predict(x1, test[0:1000, :], inputDim, filterDim, numFilters, poolDim, pooledDim, outputSize), delimiter=",", fmt="\"%d\"")
np.savetxt("../thetaCNN.csv", x1, delimiter=",", fmt="%f")
'''