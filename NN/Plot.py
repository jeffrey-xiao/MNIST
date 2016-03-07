import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def loadCSV (filePath, skip=1) :
    return np.asarray(pd.read_csv(filePath, skiprows=skip, header=None))

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

data = loadCSV('../test.csv')
theta = loadCSV('../theta.csv', 0)


m = np.size(data, 0);
featureSize = 785
hiddenSize = 100
outputSize = 10

for i in range(m):
    img = 1 - data[i, :].reshape(28, 28) / 255.0
    disp = np.zeros((28, 28, 3))
    
    disp = np.dstack((img, img, img))
    plt.imshow(disp, interpolation='nearest')
    plt.xlabel(predict(theta, data[i, :], 1, featureSize, hiddenSize, outputSize))
    plt.show()
    
