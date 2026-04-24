import numpy as np
import math
from typing import Callable
from scalar_optimization import dichotomy


a, b = 3, 2

def func(x, y):
    return 20 - ((x - a) * math.exp(-(x - a))) - ((y - b) * math.exp(-(y - b)))

def make_line_search(point: np.ndarray, direction: np.ndarray) -> Callable[[float], float]:
    return lambda lam: func(point[0] + lam * direction[0], point[1] + lam * direction[1])

def conjugate_directions_method(x0: list, eps: float) -> tuple:
    n = len(x0)
    x = np.array(x0, dtype=float)
    iter = 1
    
    print(f"Начальная точка: x^0 = {x}")
    
    while True:
        x_0 = x.copy()
        S = [np.eye(n)[i].copy() for i in range(n)]
        
        k = 1
        f_line = make_line_search(x, S[k-1])
        lam_k = dichotomy(f_line, -10.0, 10.0, eps, False)
        x = x + lam_k * S[k-1]
        print(f"  Шаг 2 (k=1): λ = {lam_k:.6f}, x = {x}")
        
        while k < n:
            y_prev = x + np.eye(n)[k]
            
            y_curr = x.copy()
            for i in range(1, k + 1):
                f_line_seq = make_line_search(y_curr, S[i-1])
                lam_opt = dichotomy(f_line_seq, -10.0, 10.0, eps, False)
                y_curr = y_curr + lam_opt * S[i-1]
                
            k += 1
            S[k-1] = x - y_prev
            
            f_line_new = make_line_search(x, S[k-1])
            lam_k = dichotomy(f_line_new, -10.0, 10.0, eps, False)
            x = x + lam_k * S[k-1]
            print(f"  Шаг 5 (k={k}): Новое S_{k} = {S[k-1]}, λ = {lam_k:.6f}, x = {x}")
            
        
        diff_norm = np.linalg.norm(x_0 - x)
        print(f"\nИтерация {iter}: ||x^0 - x_n|| = {diff_norm:.8f}")
        print(f"Текущая точка: {x}, f(x) = {func(x[0], x[1]):.8f}\n")
        
        if diff_norm <= eps:
            break
        
        iter += 1

    return x, func(x[0], x[1])

epsilon = 1e-3
x_opt, f_opt = conjugate_directions_method([0.0, 0.0], epsilon)