import numpy as np
import random
from math import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class NeuralNetworkClass:
    def __init__(self, size_network, n_epochs, s_batch, learn_r):
        self.num_epochs = n_epochs;
        self.size_batch = s_batch;
        self.learn_rate = learn_r;
        self.network_length = np.shape(size_network)[0]
        self.w = list()
        self.b = list()
        
        np.random.seed(42)

        for k in range(1, self.network_length):

            self.w.append(np.random.normal(0, 1, [size_network[k], size_network[k - 1]]))
            self.b.append(np.random.normal(0, 1, size_network[k]))

    def FeedForward(self, a):
        
        a = np.squeeze(np.asarray(a))

        for w, b in zip(self.w, self.b):
            a = self.Sigmoid(np.squeeze(np.asarray(np.dot(w, a))) + b)
        
        return a

    def Sigmoid(self, x):
    
        return 1.0/(1.0 + np.exp(-x))

    def Sigmoid_deriv(self, x):

        return np.multiply(self.Sigmoid(x), (1.0 - self.Sigmoid(x)))

    def Cost_deriv(self, act, y):

        return (act - y)

    def SGD(self, training_set, test_set):
        
        n = np.shape(training_set)[2]
        
        plot_resolution = 100

        xplot = np.linspace(0,1, plot_resolution)
        yplot = np.zeros((self.num_epochs, plot_resolution))
        
        for k in range(self.num_epochs):

            train = np.matrix(training_set[0])
            train_res = np.matrix(training_set[1])
            shuffled = np.random.permutation(n)
            shuffled = [int(i) for i in shuffled]

            
            for m in range(0, n, self.size_batch):

                train_tmp     = [np.array(training_set[0][:, k]) for k in shuffled[m:m + self.size_batch]]
                train_res_tmp = [np.array(training_set[1][:, k]) for k in shuffled[m:m + self.size_batch]]
                train_tmp     = np.stack(train_tmp, axis=1)
                train_res_tmp = np.stack(train_res_tmp, axis=1)
                train_tmp     = np.matrix(train_tmp)
                train_res_tmp = np.matrix(train_res_tmp)

                mini_batch = []

                mini_batch.append(train_tmp)
                mini_batch.append(train_res_tmp)

                self.Update_Batch(mini_batch)

            print('Epoch ', k, ' : ', self.Evaluate(test_set))

            for l in range(plot_resolution):

                yplot[k, l] = np.asscalar(self.FeedForward(xplot[l]))
        
        return xplot, yplot

    def Update_Batch(self, mini_batch):

        n_W = [np.zeros(w.shape) for w in self.w]
        n_B = [np.zeros(b.shape) for b in self.b]

        for n in range(np.shape(mini_batch)[2]):
            
            x = mini_batch[0][:, n]
            y = mini_batch[1][:, n]
            
            d_W, d_B = self.BackPropagation(x, y)

            n_W = [w + dw for w, dw in zip(n_W, d_W)]
            n_B = [b + db for b, db in zip(n_B, d_B)]

        self.w = [w - (self.learn_rate/np.shape(mini_batch)[2]) * nW for w, nW in zip(self.w, n_W)]
        self.b = [b - (self.learn_rate/np.shape(mini_batch)[2]) * nB for b, nB in zip(self.b, n_B)]

    def BackPropagation(self, x, y):
        ''' Backpropagation '''

        act = x
        acts = []
        acts.append(act)
        z_vec = []

        d_W = [np.zeros(w.shape) for w in self.w]        
        d_B = [np.zeros(b.shape) for b in self.b]
        
        for w, b in zip(self.w, self.b):
            z = np.squeeze(np.asarray(np.dot(w, act))) + b
            z_vec.append(z)
            act = self.Sigmoid(z)
            acts.append(act)

        
        delta = np.multiply(self.Cost_deriv(acts[-1], np.squeeze(np.array(y))), self.Sigmoid_deriv(z_vec[-1]))
        d_W[-1] = np.outer(delta, np.transpose(acts[-2]))
        d_B[-1] = delta
        
        for k in range(self.network_length - 2):
            z = z_vec[-k - 2]
            sig_der = self.Sigmoid_deriv(z)
            delta = np.multiply(np.dot(np.transpose(self.w[-k - 1]), delta), sig_der)
            d_W[-k - 2] = np.outer(delta, acts[-k -3].transpose())
            d_B[-k - 2] = delta
            
        return d_W, d_B
    
    def Evaluate(self, test_set):
        ''' Evaluate the neural network '''

        test_size = np.shape(test_set[0])
        test_results = np.zeros(test_size[1])

        x = np.transpose(test_set[0])
        y = np.transpose(test_set[1])

        results = np.zeros(test_size[1])
        
        for k in range(test_size[1]):
            results[k] = y[k] - self.FeedForward(x[k])
            results[k] = np.sqrt(np.sum(results[k]**2))

        return sqrt(np.sum([r**2 for r in results]))/len(results)
    
    def plotter(self, xplot, yplot, f):
        ''' Plots the output from the neural network with the reference solution '''
    
        plt.show()
        fig = plt.figure()
        ax = plt.axes(xlim = (0,1), ylim = (0,1))
        lines = []
        legends = []
    
        line, = ax.plot([], [])
        lines.append(line)
        legends.append('Neural Network')

        line, = ax.plot([], [])
        lines.append(line)
        legends.append('Analytical')
 
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(legends, loc = 1, frameon = False)
        
        y_a = f(xplot)

        def init():
            for line in lines:
                line.set_data([], [])
            return lines,

        def animate(i):
    
            for k, line in enumerate(lines):
                if k == 0:
                    line.set_data(xplot, yplot[i, :])
                else:
                    line.set_data(xplot, y_a)
            
            return lines,

        anim = animation.FuncAnimation(fig, animate, init_func = init, frames = np.shape(yplot)[0], interval = 100, blit = False)
        plt.show()
  

def func(x):
    #return 0.25 + 0.5*np.heaviside(x - 0.5, 0.5)    
    return .5 + .4*np.sin(x*2*pi) 

import time
start = time.perf_counter()

training_set = []
num_epochs = 30

trainSize = 2000
testSize = 2000

add_noise = True

trainDat = np.random.rand(1, trainSize)
testDat  = np.random.rand(1, testSize)

data_RefSet = []
data_RefAns = []
data_TestSet = []
data_TestAns = []



if add_noise:
    
    data_RefSet = trainDat
    data_RefAns = func(trainDat) - 0.01*np.random.randn(len(trainDat))
    data_TestSet = testDat
    data_TestAns = func(testDat) - 0.01*np.random.randn(len(testDat))

else:

    data_RefSet = trainDat
    data_RefAns = func(trainDat)
    data_TestSet = testDat
    data_TestAns = func(testDat)

training_set.append(data_RefSet)
training_set.append(data_RefAns)

test_set = []
test_set.append(data_TestSet)
test_set.append(data_TestAns)

net = NeuralNetworkClass([1, 20, 20, 20, 1], num_epochs, 1, 2.0)
xplot, yplot = net.SGD(training_set, test_set)

res = net.Evaluate(test_set)

end = time.perf_counter()

print('Elapsed time = ', end - start)

start = time.perf_counter()

net.FeedForward(trainDat[0, 0])

end = time.perf_counter()

print('Evaluating network time = ', end - start)

net.plotter(xplot, yplot, func)


