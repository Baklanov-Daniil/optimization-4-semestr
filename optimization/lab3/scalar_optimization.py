import math
import pandas as pd
from typing import Callable, List, Dict, Any, Optional

__all__ = ['golden_ratio', 'dichotomy', 'fibonacci']

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def _print_table(data: List[Dict[str, Any]], method_name: str, a: float, b: float, prec: int) -> None:
    """Общая функция для вывода таблицы результатов"""
    if not data:
        return
    
    df = pd.DataFrame(data)
    print("\n" + "="*100)
    print(f"{method_name.upper()} SEARCH RESULTS".center(100))
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    print(f"\nФинальный интервал: a = {round(a, prec)}, b = {round(b, prec)}")


def _calculate_precision(epsilon: float) -> int:
    """Вычисляет количество знаков после запятой для округления"""
    return int(-math.log10(epsilon)) + 2


# ==================== МЕТОД ЗОЛОТОГО СЕЧЕНИЯ ====================

def golden_ratio(func: Callable[[float], float], a: float, b: float, 
                 epsilon: float = 0.01, verbose: bool = True) -> float:
    prec = _calculate_precision(epsilon)
    data = []
    t = 0.618
    L = round(b - a, prec)
    
    x1 = round(a + L * t, prec)
    x2 = round(b - L * t, prec)
    
    f1 = round(func(x1), prec)
    f2 = round(func(x2), prec)
    
    iteration = 0
    
    if f1 > f2:
        a_new, b_new = x1, b
    else:
        a_new, b_new = a, x2
    
    if verbose:
        data.append({
            'No': iteration, 'ak': a, 'bk': b,
            'x1': x1, 'f(x1)': f1,
            'x2': x2, 'f(x2)': f2,
            'ak+1': a_new, 'bk+1': b_new
        })
    
    if f1 > f2:
        b = x1
        f1 = f2
        x1 = x2
        L = round(b - a, prec)
        x2 = round(b - L * t, prec)
        f2 = round(func(x2), prec)
    else:
        a = x2
        f2 = f1
        x2 = x1
        L = round(b - a, prec)
        x1 = round(a + L * t, prec)
        f1 = round(func(x1), prec)
    
    while L >= epsilon:
        iteration += 1
        
        if f1 > f2:
            a_new, b_new = x1, b
            b = x1
            f1 = f2
            x1 = x2
            L = round(b - a, prec)
            x2 = round(b - L * t, prec)
            f2 = round(func(x2), prec)
        else:
            a_new, b_new = a, x2
            a = x2
            f2 = f1
            x2 = x1
            L = round(b - a, prec)
            x1 = round(a + L * t, prec)
            f1 = round(func(x1), prec)
        
        if verbose:
            data.append({
                'No': iteration, 'ak': a, 'bk': b,
                'x1': x1, 'f(x1)': f1,
                'x2': x2, 'f(x2)': f2,
                'ak+1': a_new, 'bk+1': b_new
            })
    
    if verbose:
        _print_table(data, "Golden Section", a, b, prec)
    
    return round((a + b) / 2, prec - 2)


# ==================== МЕТОД ДИХОТОМИИ ====================

def dichotomy(func: Callable[[float], float], a: float, b: float,
              epsilon: float = 0.01, verbose: bool = True) -> float:
    prec = _calculate_precision(epsilon)
    data = []
    iteration = 0
    
    a_curr = round(a, prec)
    b_curr = round(b, prec)
    
    if verbose:
        data.append({
            'No': iteration, 'ak': a_curr, 'bk': b_curr,
            'c': '-', 'x1': '-', 'x2': '-', 
            'f(x1)': '-', 'f(x2)': '-',
            'ak+1': '-', 'bk+1': '-'
        })
    
    while (b_curr - a_curr) > (2 * epsilon):
        iteration += 1
        c = round((a_curr + b_curr) / 2, prec)
        x1 = round(c - epsilon / 2, prec)
        x2 = round(c + epsilon / 2, prec)
        f1 = round(func(x1), prec)
        f2 = round(func(x2), prec)
        
        if f1 > f2:
            a_new, b_new = x1, b_curr
            a_curr = x1
        else:
            a_new, b_new = a_curr, x2
            b_curr = x2
        
        if verbose:
            data.append({
                'No': iteration,
                'ak': a_curr,
                'bk': b_curr,
                'c': c,
                'x1': x1,
                'x2': x2,
                'f(x1)': f1,
                'f(x2)': f2,
                'ak+1': a_new,
                'bk+1': b_new
            })
    
    if verbose:
        _print_table(data, "Dichotomy", a_curr, b_curr, prec)
    
    return round((a_curr + b_curr) / 2, prec - 2)


# ==================== МЕТОД ФИБОНАЧЧИ ====================

def _generate_fibonacci_sequence(initial_length: float, epsilon: float) -> List[int]:
    fib = [0, 1]
    while fib[-1] < (initial_length / epsilon):
        fib.append(fib[-1] + fib[-2])
    return fib


def fibonacci(func: Callable[[float], float], a: float, b: float,
              epsilon: float = 0.01, verbose: bool = True) -> float:
    prec = _calculate_precision(epsilon)
    data = []
    
    a_curr = round(a, prec)
    b_curr = round(b, prec)
    
    fib = [1, 1]
    while fib[-1] < (b_curr - a_curr) / epsilon:
        fib.append(fib[-1] + fib[-2])
    
    N = len(fib)
    n = N - 2
    
    L = round(b_curr - a_curr, prec)
    delta = round(L / fib[N-1], prec)
    
    x1 = round(a_curr + fib[N-3] * delta, prec)
    x2 = round(b_curr - fib[N-3] * delta, prec)
    
    f1 = round(func(x1), prec)
    f2 = round(func(x2), prec)
    
    iteration = 0
    
    if f1 > f2:
        a_new, b_new = x1, b_curr
    else:
        a_new, b_new = a_curr, x2
    
    if verbose:
        data.append({
            'No': iteration, 'ak': a_curr, 'bk': b_curr,
            'x1': x1, 'f(x1)': f1,
            'x2': x2, 'f(x2)': f2,
            'ak+1': a_new, 'bk+1': b_new
        })
    
    for k in range(1, n):
        iteration += 1
        
        if f1 > f2:
            a_curr = x1
            x1 = x2
            f1 = f2
            L = round(b_curr - a_curr, prec)
            delta = round(L / fib[N - k - 1], prec)
            x2 = round(b_curr - fib[N - k - 3] * delta, prec)
            f2 = round(func(x2), prec)
        else:
            b_curr = x2
            x2 = x1
            f2 = f1
            L = round(b_curr - a_curr, prec)
            delta = round(L / fib[N - k - 1], prec)
            x1 = round(a_curr + fib[N - k - 3] * delta, prec)
            f1 = round(func(x1), prec)
        
        if f1 > f2:
            a_new, b_new = x1, b_curr
        else:
            a_new, b_new = a_curr, x2
        
        if verbose:
            data.append({
                'No': iteration, 'ak': a_curr, 'bk': b_curr,
                'x1': x1, 'f(x1)': f1,
                'x2': x2, 'f(x2)': f2,
                'ak+1': a_new, 'bk+1': b_new
            })
    
    iteration += 1
    if f1 > f2:
        b_curr = x1
    else:
        a_curr = x2
    
    if verbose:
        data.append({
            'No': iteration, 'ak': a_curr, 'bk': b_curr,
            'x1': '-', 'f(x1)': '-',
            'x2': '-', 'f(x2)': '-',
            'ak+1': '-', 'bk+1': '-'
        })
        _print_table(data, "Fibonacci", a_curr, b_curr, prec)
    
    return round((a_curr + b_curr) / 2, prec - 2)