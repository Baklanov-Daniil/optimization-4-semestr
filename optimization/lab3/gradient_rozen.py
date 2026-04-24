import math
import numpy as np
from scalar_optimization import golden_ratio

print("Метод проекции градиента Розена для функции f(x) = x1^2 - x2^2\nОграничение: ϕ(x) = x1 +4x2 - 3 = 0\n")

a = 3
A = np.array([[1, 4]])

def f(x):
    return x[0]**2 - x[1]**2

def grad_f(x):
    return np.array([2 * x[0], -2 * x[1]])

def phi(x):
    return x[0] + 4 * x[1] - a

I = np.eye(2) 
P = I - A.T @ np.linalg.inv(A @ A.T) @ A # матрица проекции P

# x0
x = np.array([3.0, 0.0])
epsilon = 1e-2
f_prev = f(x)

for k in range(1, 101):
    # п3
    S = -P @ grad_f(x)
    S_norm = np.linalg.norm(S)
    
    # п6
    func_ = lambda alpha: f(x + alpha * S)
    alpha = golden_ratio(func_, 0, 5, 1e-2, False)
    alpha = min(alpha, 2.0)
    
    x_new = x + alpha * S
    x_new[0] = a - 4 * x_new[1]
    
    f_new = f(x_new)
    f_diff = abs(f_prev - f_new)
    step_norm = np.linalg.norm(x_new - x)
    
    print(f"Итерация {k}: x1 = {x_new[0]}, x2 = {x_new[1]}, f(x) = {f_new}, ||s|| = {S_norm}, Δf = {f_diff:<6.6e}")
    
    # п4
    if S_norm <= epsilon:
        # print("Стоп по направлению")
        break
    
    if f_diff < 1e-6:
        # print("Стоп по функции")
        break
    
    if step_norm < 1e-6:
        # print("Стоп по шагу")
        break
    
    x = x_new
    f_prev = f_new

print(f"\nx* = ({x[0]}, {x[1]})")
print(f"f(x*) = {f(x)}")
print(f"φ(x*) = {phi(x)}")
print(f"Итераций: k = {k}")