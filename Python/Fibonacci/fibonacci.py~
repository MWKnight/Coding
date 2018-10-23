import numpy as np

def Fibonacci(n):
    num = np.zeros(n)
    
    if n > 0:
        num[0] = 1

    if n > 1:
        num[1] = 1

    if n > 2:

        for k in range(2, n):
            num[k] = num[k - 1] + num[k - 2]

    return num

m = 10
fib = Fibonacci(m)

for f in range(m):
    print(f+1, ':', fib[f])
