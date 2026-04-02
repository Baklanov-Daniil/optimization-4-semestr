import math
import numpy as np
from scalar_optimization import golden_ratio
# пункт 1
epsilon = 1e-2
x_current = np.zeros(2)
a = 3
b = 2
max_iter = 50
lambda_search = np.array([0, 5])
# f (x, y) = A - (x - a) * e ** -(x - a) - (y - b) * e ** -(x - b);   A = 20, a = 3, b = 2
def func(x, y):
    return 20 - ((x - a) * math.exp(-(x - a))) - ((y - b) * math.exp(-(y - b)))

def gradient(x, y):
    grad_x = math.exp(-(x - a)) * (x - a - 1)
    grad_y = math.exp(-(y - b)) * (y - b - 1)
    return np.array([grad_x, grad_y])

x_k = x_current.copy()
k = 1
f_prev = func(x_k[0], x_k[1])

while k < max_iter:
    # пункт 2
    grad = gradient(x_k[0], x_k[1])
    grad_norm = np.linalg.norm(grad)
    # пункт 3
    if grad_norm < epsilon:
        break
    
    s_k = -(grad / grad_norm)
    # пункт 4
    func_ = lambda lam: func(x_k[0] + lam * s_k[0], x_k[1] + lam * s_k[1])
    
    lambda_k = golden_ratio(func_, lambda_search[0], lambda_search[1], epsilon = epsilon, verbose = False)
    
    x_new = x_k + lambda_k * s_k
    
    print(f"\nk (Итерация) = {k} | λ = {lambda_k} | grad_norm = {grad_norm} | x_new (x_k) = {x_new}")
       
    x_k = x_new
    # пункт 5
    k += 1

print(f"\nx* = ({x_k[0]:.6f}, {x_k[1]:.6f}), f(x*) = {func(x_k[0], x_k[1]):.6f}, k = {k - 1}")