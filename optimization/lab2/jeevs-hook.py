import math
import numpy as np
from scalar_optimization import golden_ratio

# пункт 1
epsilon = 1e-2
x_current = np.zeros(2)
a = 3
b = 2
max_iter = 50
delta = 0.5  # начальный шаг исследования
# f (x, y) = A - (x - a) * e ** -(x - a) - (y - b) * e ** -(y - b);   A = 20, a = 3, b = 2
def func(x, y):
    return 20 - ((x - a) * math.exp(-(x - a))) - ((y - b) * math.exp(-(y - b)))

x_k = x_current.copy()
x_base = x_k.copy()
k = 1

while k < max_iter:
    # пункт 2 - Исследовательский поиск
    f_base = func(x_base[0], x_base[1])
    x_new = x_base.copy()
    
    for i in range(2):  # по каждой координате
        x_temp = x_new.copy()
        x_temp[i] += delta
        f_temp = func(x_temp[0], x_temp[1])
        
        if f_temp < f_base:  # пункт 3 - шаг вперёд успешен
            x_new[i] = x_temp[i]
            f_base = f_temp
        else:
            x_temp[i] -= 2 * delta  # пункт 4 - шаг назад
            f_temp = func(x_temp[0], x_temp[1])
            
            if f_temp < f_base:
                x_new[i] = x_temp[i]
                f_base = f_temp
    
    # пункт 5 - Ускорение (pattern move)
    if np.linalg.norm(x_new - x_base) > epsilon:
        x_pattern = x_base + 2 * (x_new - x_base)
        f_pattern = func(x_pattern[0], x_pattern[1])
        
        if f_pattern < f_base:
            x_base = x_pattern.copy()
        else:
            x_base = x_new.copy()
    else:
        x_base = x_new.copy()
    
    print(f"\nk (Итерация) = {k} ; delta = {delta} ; x_new (x_k) = {x_base}")
    
    # пункт 6 - Проверка сходимости
    if np.linalg.norm(x_base - x_k) < epsilon:
        delta *= 0.5  # уменьшаем шаг
        if delta < epsilon:
            break
    
    x_k = x_base.copy()
    k += 1

print(f"\nx* = ({x_k[0]:.6f}, {x_k[1]:.6f}), f(x*) = {func(x_k[0], x_k[1]):.6f}, k = {k - 1}")