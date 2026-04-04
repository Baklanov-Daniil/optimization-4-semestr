import math
import numpy as np
from scalar_optimization import dichotomy

epsilon = 1e-3
x_current = np.zeros(2)
a, b = 3, 2
max_iter = 200

def func(x, y):
    return 20 - (x - a) * math.exp(-(x - a)) - (y - b) * math.exp(-(y - b))

x_k = x_current.copy()
D = np.eye(2)
k = 1
star_x = x_k.copy()

while k <= max_iter:
    x_start = x_k.copy()
    f_start = func(x_start[0], x_start[1])
    
    for i in range(2):
        f_line = lambda alpha: func(x_k[0] + alpha * D[i][0], x_k[1] + alpha * D[i][1])
        alpha_opt = dichotomy(f_line, -10.0, 10.0, 1e-8, False)
        x_k = x_k + alpha_opt * D[i]
        
    delta = x_k - x_start
    f_new = func(x_k[0], x_k[1])
    
    print(f"k = {k} | x = {x_k} | ||Δx|| = {np.linalg.norm(delta):.2e} | Δf = {abs(f_new - f_start)}")
    
    if np.linalg.norm(delta) < epsilon or abs(f_new - f_start) < epsilon:
        star_x = x_k.copy()
        break
        
    norm_delta = np.linalg.norm(delta)
    if norm_delta > 1e-10: 
        D[0] = delta / norm_delta
        D[1] = D[1] - np.dot(D[1], D[0]) * D[0]
        norm_D1 = np.linalg.norm(D[1])
        D[1] /= norm_D1
            
    k += 1

print(f"x* = ({star_x[0]:.6f}, {star_x[1]:.6f})")
print(f"f(x*) = {func(star_x[0], star_x[1]):.6f}")
print(f"Итераций: {k}")