import numpy as np
import matplotlib.pyplot as plt

data_file = open('data.dat')

data_matrix = []


for line in data_file:
    dat = line.split(',')
    dat.pop()
    dat = [int(d) for d in dat]
    data_matrix.append(dat)


plt.imshow(data_matrix, cmap = 'RdGy')
#plt.contourf(data_matrix, cmap = 'terrain')
plt.show()

