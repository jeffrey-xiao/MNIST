import numpy as np
import scipy
import csv

def loadCSV (filePath, datatype = np.int) :
    with open(filePath,'r') as dest_file:
        data_iter = csv.reader(dest_file, delimiter=",", quotechar='"')
        data = [line for line in data_iter][1:]
    return np.asarray(data, dtype=datatype)


data = loadCSV('../train.csv')
reg = 3.0
hiddenLayerSize = 30

def sigmoid (x) :
    return 1.0 / (1 + np.exp(-x))

# args = (data)
# x: (785 x 29, 30 x 10)
def getCost (x):
    hiddenLayerTheta = x[0 : 785 * (hiddenLayerSize - 1)].reshape((785, (hiddenLayerSize - 1)))
    outputLayerTheta = x[785 * (hiddenLayerSize - 1) : 785 * (hiddenLayerSize - 1) + hiddenLayerSize * 10].reshape((hiddenLayerSize, 10))
    
    # 42000 x 1
    labels = data[:, 0]
    expectedValues = np.eye(10)[labels]
    
    # 42000 x 785
    features = np.ones((42000, 785))
    features[:, 1:] = data[:, 1:]

    # 42000 x 30
    hiddenLayer = np.ones((42000, hiddenLayerSize))
    hiddenLayer[:, 1:] = sigmoid(features.dot(hiddenLayerTheta))
    
    # 42000 x 10
    outputLayer = sigmoid(hiddenLayer.dot(outputLayerTheta))
    
    J = -np.sum(expectedValues * np.log(outputLayer) + (1 - expectedValues) * np.log(1 - outputLayer)) / 42000
    J += reg / (2 * 42000) * (np.sum(hiddenLayerTheta[:, 1:] ** 2) + np.sum(outputLayerTheta[:, 1:] ** 2))
    
    return J

print getCost(np.random.rand(785 * (hiddenLayerSize - 1) + hiddenLayerSize * 10))