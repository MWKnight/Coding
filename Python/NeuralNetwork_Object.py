import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation
import math
class NeuralNetworkClass:
    def __init__(self, sizes):

        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = []
        self.weights = []
        np.random.seed(42)

        for k in range(0, self.numLayers - 1):
            self.biases.append(np.random.normal(0, 1, self.sizes[k + 1]))
            self.weights.append(np.random.normal(0, 1, [self.sizes[k + 1], self.sizes[k]]))

    def Feedforward(self, a):
        a = np.squeeze(np.array(a))
        for k in range(0, self.numLayers - 1):

                b = self.biases[k]
                w = self.weights[k]
                a = self.sigmoid(np.squeeze(np.asarray(np.dot(w, a))) + b)

        return a

    def sigmoid(self, z):

         return 1.0/(1.0 + np.exp(-z))

    def sigmoid_prime(self, z):

         return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        ''' Stochastic Gradient Descent'''

        n = np.shape(training_data[0])[1]

        x_plot = np.linspace(0,1,100)
        y_plot = np.zeros((epochs, 100))

        for k in range(0, epochs):

                train = np.matrix(training_data[0])
                train_res = np.matrix(training_data[1])
                shuffled = np.random.permutation(n)
                shuffled = [int(i) for i in shuffled]
                print('shuffled = ', shuffled)
                for l in range(0, n, mini_batch_size):

                        train_tmp     = [np.array(training_data[0][:, k]) for k in shuffled[l:l + mini_batch_size]]
                        train_res_tmp = [np.array(training_data[1][:, k]) for k in shuffled[l:l + mini_batch_size]]
                        train_tmp     = np.stack(train_tmp, axis=1)
                        train_res_tmp = np.stack(train_res_tmp, axis=1)
                        train_tmp     = np.matrix(train_tmp)
                        train_res_tmp = np.matrix(train_res_tmp)

                        mini_batch = []

                        mini_batch.append(train_tmp)
                        mini_batch.append(train_res_tmp)
:
                        self.update_mini_batch(mini_batch, eta)

                if test_data:

                        for m in range(100):
                            y_plot[k, m] = self.Feedforward(x_plot[m])

                        #res = self.evaluate(test_data)
                        print('epoch ', k)

                        #print('accuracy = ', res)
        self.plotter(x_plot, y_plot)

    def update_mini_batch(self, mini_batch, eta):

            nabla_b = []
            nabla_w = []
            print(np.random.randn(1))
            exit()
            for k in range(0, self.numLayers - 1):
                    nabla_b.append(np.zeros((np.shape(self.biases[k]))))
                    nabla_w.append(np.zeros((np.shape(self.weights[k]))))

            mini_batch_size = np.shape(mini_batch)

            for n in range(0, mini_batch_size[2]):

                    x = mini_batch[0][:, n]
                    y = mini_batch[1][:, n]
                    print(x)
                    print(y)
                    exit()
                    [delta_nabla_b, delta_nabla_w] = self.backprop(x, y)
                    for l in range(0, self.numLayers - 1):

                            nabla_b[l] = nabla_b[l] + delta_nabla_b[l]
                            nabla_w[l] = nabla_w[l] + delta_nabla_w[l]
   #         print(eta)
  #          print(len(mini_batch))
 #           print(np.shape(mini_batch)[2])
#            exit()
            for m in range(0, self.numLayers - 1):

                    self.biases[m]  = self.biases[m]  - (eta/mini_batch_size[2]) * nabla_b[m]
                    self.weights[m] = self.weights[m] - (eta/mini_batch_size[2]) * nabla_w[m]

    def backprop(self, x, y):

            nabla_b = []
            nabla_w = []
            for k in range(0, self.numLayers - 1):
                    nabla_b.append(np.zeros((np.shape(self.biases[k]))))
                    nabla_w.append(np.zeros((np.shape(self.weights[k]))))

            act = x
            acts = []
            acts.append(act)
            zs = []
            for k in range(0, self.numLayers - 1):
                    b = self.biases[k]
                    w = self.weights[k]
                    #act = np.array(np.squeeze(np.asarray(act)))
                    z = np.squeeze(np.asarray(np.dot(w, act))) + b
                    zs.append(z)
                    #print('w*a =', np.shape(np.dot(w, act)))
                    #print('z = ', np.shape(z))
                    #print('w = ', np.shape(w))
                    #print('b = ', np.shape(b))
                    #print('act = ', np.shape(act))
                    act = self.sigmoid(z)
                    acts.append(act)

            delta       = np.multiply(self.cost_derivative(acts[-1], np.squeeze(np.array(y))), self.sigmoid_prime(zs[-1]))
            nabla_b[-1] = delta
            nabla_w[-1] = np.outer(delta, np.transpose(acts[-2]))
            #print('act', np.shape(np.transpose(acts[-2])))
            #print('delta = ', np.shape(delta))
            #print('nabla_w[-1] =', np.shape(nabla_w[-1]))

            for l in range(0, self.numLayers - 2):
                    z = zs[-l - 2]
                    sp = self.sigmoid_prime(z)
                    test = np.dot(np.transpose(self.weights[-l - 1]), delta)
                    delta = np.multiply(np.dot(np.transpose(self.weights[-l - 1]), delta), sp)
                    nabla_b[-l - 2] = delta
                    nabla_w[-l - 2] = np.outer(delta, np.transpose(acts[-l -3]))


            return nabla_b, nabla_w

    def evaluate(self, test_data):

            test_size = np.shape(test_data[0])
            test_results = np.zeros(test_size[1])

            x = np.transpose(test_data[0])
            y = np.transpose(test_data[1])
            out = np.zeros(test_size[1])

            for k in range(0, test_size[1]):

                    m = np.argmax(self.Feedforward(np.squeeze(x[k, :])))
                    m2 = np.argmax(np.squeeze(y[k, :]))
                    test_results[k] = (m - m2)
                    out[k] = (m == m2)

            return sum(out)/test_size[1]

    def cost_derivative(self, output_acts, y):

            return output_acts - y

    def plotter(self, xplot, yplot):

        def func(x):

            y = .5 + .4*np.sin(x*2*math.pi)

            return y

        plt.show()
        fig = plt.figure()
        ax = plt.axes(xlim=(0, 1), ylim=(0, 1))

        lines=[]     # list for plot lines for solvers and analytical solutions
        legends=[]   # list for legends for solvers and analytical solutions


        line, = ax.plot([], [])
        lines.append(line)
        legends.append('Neural Network')

        line, = ax.plot([], []) #add extra plot line for analytical solution
        lines.append(line)
        legends.append('Analytical')

        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend(legends, loc=1, frameon=False)

        # initialization function: plot the background of each frame
        def init():
            for line in lines:
                line.set_data([], [])
            return lines,

        # animation function.  This is called sequentially
        #print(np.shape(yplot))
        #print(np.shape(yplot)[0])
        #exit()
        def animate(i):
            for k, line in enumerate(lines):
                if (k==0):
                    line.set_data(xplot, yplot[i, :])
                else:
                    line.set_data(xplot, func(xplot))
            return lines,

        # call the animator.  blit=True means only re-draw the parts that have changed.

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=np.shape(yplot)[0], interval=100, blit=False)

        plt.show()
