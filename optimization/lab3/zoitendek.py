import math
import numpy as np
from scalar_optimization import golden_ratio

a = 3

print("Метод Зойтендека для функции f(x) = x1^2 - x2^2\nОграничение: ϕ(x) = x1 +4x2 - 3 = 0\n")

def f(x):
    return x[0]**2 - x[1]**2

def grad_f(x):
    return np.array([2 * x[0], -2 * x[1]])

def phi(x):
    return x[0] + 4 * x[1] - a

def grad_phi(x):
    return np.array([1, 4])

def feasible_direction(x):
    g = grad_f(x)
    h = grad_phi(x)
    s = -g - (np.dot(-g, h) / np.dot(h, h)) * h
    if np.linalg.norm(s) < 1e-8:
        return np.zeros(2)
    return s / np.linalg.norm(s)

x = np.array([3.0, 0.0])

for k in range(1, 101):
    s = feasible_direction(x)
    s_norm = np.linalg.norm(s)
    
    print(f"Итерация {k}: x1 = {x[0]}, x2 = {x[1]}, f(x) = {f(x):<12.6f}, ||s|| = {s_norm:<6.6e},  φ(x) = {phi(x):<6.6e}")
    
    if s_norm < 1e-6:
        #print("Направление нулевое")
        break
    
    func_ = lambda lam: f(x + lam * s)
    lam = golden_ratio(func_, 0, 5, 1e-2, False)
    lam = min(lam, 2.0)
    
    x_new = x + lam * s
    x_new[0] = a - 4 * x_new[1]  # проекция на ограничение
    
    if np.linalg.norm(x_new - x) < 1e-6:
        #print("Шаг слишком мал")
        break
    x = x_new

print(f"\nx* = ({x[0]}, {x[1]})")
print(f"f(x*) = {f(x)}")
print(f"φ(x*) = {phi(x)}")
print(f"Итераций: k = {k}")