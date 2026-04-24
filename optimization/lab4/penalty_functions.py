import numpy as np
from scipy.optimize import minimize

epsilon = 1e-4
x0 = np.array([2.0, 2.0])
r = 10.0
c = 10.0
max_outer_iter = 30

def f(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2

def h(x):
    return x[0] - 2*x[1] + 1

def g(x):
    return 1 - 0.25*x[0]**2 - x[1]**2

def penalty(x):
    penalty_ineq = (max(0, -g(x)))**2
    penalty_eq = (h(x))**2
    return penalty_ineq + penalty_eq

def F(x, r):
    return f(x) + r * penalty(x)

print("Метод штрафных функций для f(x) = (x1 - 2)^2 + (x2 - 1)^2\n")
print("Ограничения:\nx1 - 2x2 + 1 = 0\n0.25x1^2 + x2^2 <= 1\n\nx0 = (2, 2), epsilon = 0.01\n")

x_k = x0.copy()

for k in range(1, max_outer_iter + 1):
    result = minimize(lambda x: F(x, r), x_k, method='BFGS')
    x_new = result.x
    
    alpha = penalty(x_new)
    g_val = g(x_new)
    
    print(f"Итерация {k}: x1 = {x_new[0]:.6f} ; x2 = {x_new[1]:.6f} ; f(x) = {f(x_new):.6f} ; a(x) = {alpha} ; r = {r} ; g(x) = {g_val}")
    
    if alpha < epsilon and g_val >= -1e-6:
        print(f"\na(x) < epsilon\nМетод окончен")
        break
    
    r *= c
    x_k = x_new

print(f"\nРешение: x* = ({x_new[0]:.6f}, {x_new[1]:.6f})")
print(f"f(x*) = {f(x_new):.6f}")
print(f"h(x) = {h(x_new):.6f}")
print(f"g(x) = {g(x_new):.6f}")
print(f"0.25x1² + x2² = {0.25*x_new[0]**2 + x_new[1]**2:.6f}")
