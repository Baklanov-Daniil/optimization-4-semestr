import math
import numpy as np
from scalar_optimization import golden_ratio

epsilon = 1e-6
x_current = np.zeros(2)
lambda_search = np.array([-10, 10])
a = 3
b = 2
max_iter = 100
# f (x, y) = A - (x - a) * e ** -(x - a) - (y - b) * e ** -(x - b);   A = 20, a = 3, b = 2
def func(x, y):
    return 20 - ((x - a) * (math.exp(-(x - a)))) - ((y - b) * (math.exp(-(y - b))))

x_k = x_current.copy()
k = 1

while k < max_iter:
    y = x_k.copy()
    j = 1
    
    while True:
        if j == 1: # пункт 1
            func_ = lambda lambda_: func(y[0] + lambda_, y[1]) # движение по e1
            lambda_j = golden_ratio(func_, lambda_search[0], lambda_search[1], epsilon = epsilon, verbose = False)
            y[0] += lambda_j
        else:
            func_ = lambda lambda_: func(y[0], y[1] + lambda_) # движение по e2
            lambda_j = golden_ratio(func_, lambda_search[0], lambda_search[1], epsilon = epsilon, verbose = False)
            y[1] += lambda_j
        # пункт 2
        if j < 2: 
            j += 1
        else: 
            break
    
    x_new = y.copy() # пункт 3
    print(f"\nk (Итерация) = {k} | λ = {lambda_j} | x_new (x_k) = {x_new} | j = {j}")
    
    if (np.linalg.norm(x_new - x_k) < epsilon):
        star_x = x_new
        break
    else:
        x_k = x_new
        j = 1
        k += 1
        
print(f"\nx* = ({star_x[0]:.6f}, {star_x[1]:.6f})\nf(x*) = {func(star_x[0], star_x[1]):.6f}")