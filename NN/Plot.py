import matplotlib.pyplot as plt
import numpy as np
import csv

def loadCSV (filePath, datatype = np.int) :
    with open(filePath,'r') as dest_file:
        data_iter = csv.reader(dest_file, delimiter=",", quotechar='"')
        data = [line for line in data_iter][1:]
    return np.asarray(data, dtype=datatype)

# data = loadCSV('../train.csv')
data = loadCSV('../test.csv')

for i in range(42000):
    img = 1 - data[i][:].reshape(28, 28) / 255.0
    disp = np.zeros((28, 28, 3))
    
    disp = np.dstack((img, img, img))
    plt.imshow(disp, interpolation='nearest')
    plt.show()
