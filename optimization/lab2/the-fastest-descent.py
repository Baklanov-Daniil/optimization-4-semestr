# Наискорейший спуск (3 метод)
import math
import numpy as np
from scalar_optimization import golden_ratio

epsilon = 1e-6
x_current = np.zeros(2)
a = 3
b = 2
max_iter = 10000
lambda_search = np.array([-10, 10])

# f (x, y) = A - (x - a) * e ** -(x - a) - (y - b) * e ** -(x - b);   A = 20, a = 3, b = 2
def func(x, y):
    return 20 - ((x - a) * (math.exp(-(x - a)))) - ((y - b) * (math.exp(-(y - b))))

def gradient(x, y): # производную по x и y посчитал на листке
    grad_x = math.exp(-(x - a)) * (x - a - 1)
    grad_y = math.exp(-(y - b)) * (y - b - 1)
    return np.array([grad_x, grad_y])

# пункт 1
x_k = x_current.copy()
k = 1 

while k < max_iter:
    print(f"Итерация {k} ...")
    grad = gradient(x_k[0], x_k[1])
    grad_norm = np.linalg.norm(grad) # - пункт 2
    
    if grad_norm < epsilon: # пункт 3
        star_x = x_k
        break
    else:
        s_k = -(grad / grad_norm)
    
    # пункт 4
    func_ = lambda lambda_: func(x_k[0] + lambda_ * s_k[0], x_k[1] + lambda_ * s_k[1])
    lambda_k = golden_ratio(func_, lambda_search[0], lambda_search[1], epsilon = epsilon, verbose = False)
    # пункт 5
    k += 1

print(f"x* = ({star_x[0], star_x[1]}); f(x*) = {func(star_x[0], star_x[1])}")
    