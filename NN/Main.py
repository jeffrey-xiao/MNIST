import numpy as np
from scipy.optimize import *
import csv

def loadCSV (filePath, datatype = np.int) :
    with open(filePath,'r') as dest_file:
        data_iter = csv.reader(dest_file, delimiter=",", quotechar='"')
        data = [line for line in data_iter][1:]
    return np.asarray(data, dtype=datatype)


data = loadCSV('../train.csv')
test = loadCSV('../test.csv')
reg = 3.0
hiddenLayerSize = 50
Nfeval = 1

def sigmoid (x) :
    return 1.0 / (1 + np.exp(-x))

def sigmoidGrad (x) :
    return sigmoid(x) * (1 - sigmoid(x))

# args = (data)
# x: (785 x 29, 30 x 10)

def predict (x, m) :
    hiddenLayerTheta = x[0 : 785 * (hiddenLayerSize - 1)].reshape((785, (hiddenLayerSize - 1)))
    outputLayerTheta = x[785 * (hiddenLayerSize - 1) : 785 * (hiddenLayerSize - 1) + hiddenLayerSize * 10].reshape((hiddenLayerSize, 10))
    
    # 42000 x 785
    features = np.ones((m, 785))
    features[:, 1:] = test / 255.0

    # 42000 x 30
    hiddenLayer = np.ones((m, hiddenLayerSize))
    hiddenLayer[:, 1:] = sigmoid(features.dot(hiddenLayerTheta))
    
    # 42000 x 10
    outputLayer = sigmoid(hiddenLayer.dot(outputLayerTheta))
    
    return np.argmax(outputLayer, 1)

def getCost (x):
    hiddenLayerTheta = x[0 : 785 * (hiddenLayerSize - 1)].reshape((785, (hiddenLayerSize - 1)))
    outputLayerTheta = x[785 * (hiddenLayerSize - 1) : 785 * (hiddenLayerSize - 1) + hiddenLayerSize * 10].reshape((hiddenLayerSize, 10))
    
    # 42000 x 1
    labels = data[:, 0]
    expectedValues = np.eye(10)[labels]
    
    # 42000 x 785
    features = np.ones((42000, 785))
    features[:, 1:] = data[:, 1:] / 255.0

    # 42000 x 30
    hiddenLayer = np.ones((42000, hiddenLayerSize))
    hiddenLayer[:, 1:] = sigmoid(features.dot(hiddenLayerTheta))
    
    # 42000 x 10
    outputLayer = sigmoid(hiddenLayer.dot(outputLayerTheta))
    
    J = -np.sum(expectedValues * np.log(outputLayer + 0.000001) + (1 - expectedValues) * np.log(1 - outputLayer + 0.000001)) / 42000
    J += reg / (2 * 42000) * (np.sum(hiddenLayerTheta[:, 1:] ** 2) + np.sum(outputLayerTheta[:, 1:] ** 2))
    return J

def getGradient (x) :
    # 785 x 29
    hiddenLayerTheta = x[0 : 785 * (hiddenLayerSize - 1)].reshape((785, (hiddenLayerSize - 1)))
    
    # 30 x 10
    outputLayerTheta = x[785 * (hiddenLayerSize - 1) : 785 * (hiddenLayerSize - 1) + hiddenLayerSize * 10].reshape((hiddenLayerSize, 10))
    
    # 42000 x 1
    labels = data[:, 0]
    expectedValues = np.eye(10)[labels]
    
    # 42000 x 785
    features = np.ones((42000, 785))
    features[:, 1:] = data[:, 1:] / 255.0

    # 42000 x 30
    hiddenLayer = np.ones((42000, hiddenLayerSize))
    hiddenLayer[:, 1:] = sigmoid(features.dot(hiddenLayerTheta))
    
    # 42000 x 10
    outputLayer = sigmoid(hiddenLayer.dot(outputLayerTheta))
    # print outputLayer[0].tolist()
    # print outputLayer[1].tolist()
    
    # print "\n"
    # 42000 x 10
    err3 = (outputLayer - expectedValues) *1 #sigmoidGrad(hiddenLayer.dot(outputLayerTheta))
    
    # 42000 x 29
    err2 = err3.dot(outputLayerTheta.transpose())[:, 1:] * sigmoidGrad(features.dot(hiddenLayerTheta))
    
    hiddenLayerTheta[:, 0] = 0
    outputLayerTheta[:, 0] = 0
    
    # 30 x 10
    grad2 = hiddenLayer.transpose().dot(err3) / 42000 + reg * outputLayerTheta / 42000
    
    # 785 x 29
    grad1 = features.transpose().dot(err2) / 42000 + reg * hiddenLayerTheta / 42000
    # print np.concatenate((grad1.flatten(), grad2.flatten()), axis=0)
    return np.concatenate((grad1.flatten(), grad2.flatten()), axis=0)



def callback (xk) :
    global Nfeval
    print Nfeval
    Nfeval += 1

print "Before learning"
x0 = np.random.rand(785 * (hiddenLayerSize - 1) + hiddenLayerSize * 10) / 100
x1 = optimize.fmin_cg(getCost, x0, fprime = getGradient, callback=callback, maxiter=200)
print predict(x1, 28000)
np.savetxt("../results.csv", predict(x1, 28000), delimiter=",", fmt="\"%d\"")