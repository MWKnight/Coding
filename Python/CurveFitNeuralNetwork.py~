import numpy as np
import random
import math
from NeuralNetwork_Object import NeuralNetworkClass

def func(x):

    y = .5 + .4*np.sin(x*2*math.pi)

    return y

trainSize = 1000
testSize = 1000
trainDat = np.random.rand(1, trainSize)
testDat  = np.random.rand(1, testSize)

data_RefSet = []
data_RefAns = []
data_TestSet = []
data_TestAns = []

data_RefSet = trainDat
data_RefAns = func(trainDat)
data_TestSet = testDat
data_TestAns = func(testDat)

training_data = []
#data_RefSet = np.matrix(data_RefSet)
#data_RefAns = np.matrix(data_RefAns)
#data_TestSet = np.matrix(data_TestSet)
#data_TestAns = np.matrix(data_TestAns)

#data_RefSet = np.transpose(data_RefSet)
#data_RefAns = np.transpose(data_RefAns)
#data_TestSet = np.transpose(data_TestSet)
#data_TestAns = np.transpose(data_TestAns)

training_data.append(data_RefSet)
training_data.append(data_RefAns)

print(np.shape(training_data[0]))

test_data = []
test_data.append(data_TestSet)
test_data.append(data_TestAns)

NN = NeuralNetworkClass([1, 10, 10, 10, 1])

NN.SGD(training_data, 100, 1, 2.0, True)

res = NN.evaluate(test_data)

print('Result: ', res)
