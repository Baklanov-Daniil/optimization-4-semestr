from sympy import symbols, diff, solve

def find_global_extremum(f, phi):
    x1, x2, lam = symbols('x1 x2 lambda')
    
    print("\nШАГ 1: Стационарные точки внутри области")
    df_dx1 = diff(f, x1)
    df_dx2 = diff(f, x2)
    
    stationary = solve([df_dx1, df_dx2], (x1, x2), dict=True)
    
    for sol in stationary:
        point = (sol[x1], sol[x2])
        f_val = f.subs(sol)
        phi_val = phi.subs(sol)
        print(f"Точка: {point}, f = {f_val}, φ = {phi_val}")
    
    print("\nШАГ 2: Экстремумы на границе")
    L = f + lam * phi
    
    dL_dx1 = diff(L, x1)
    dL_dx2 = diff(L, x2)
    dL_dlam = diff(L, lam)
    
    boundary = solve([dL_dx1, dL_dx2, dL_dlam], (x1, x2, lam), dict=True)
    
    for sol in boundary:
        point = (sol[x1], sol[x2])
        f_val = f.subs({x1: sol[x1], x2: sol[x2]})
        print(f"Точка: {point}, f = {f_val}, λ = {sol[lam]}")
    
    print("\nШАГ 3: Глобальные экстремумы")
    
    all_points = []
    for sol in stationary:
        all_points.append(('внутри', (sol[x1], sol[x2]), f.subs(sol)))
    for sol in boundary:
        all_points.append(('граница', (sol[x1], sol[x2]), f.subs({x1: sol[x1], x2: sol[x2]})))
    
    if all_points:
        min_pt = min(all_points, key=lambda x: x[2])
        max_pt = max(all_points, key=lambda x: x[2])
        
        print(f"Минимум: {min_pt[1]} -> f = {min_pt[2]} ({min_pt[0]})")
        print(f"Максимум: {max_pt[1]} -> f = {max_pt[2]} ({max_pt[0]})")
    
    if all_points:
        min_pt = min(all_points, key=lambda x: x[2])
        max_pt = max(all_points, key=lambda x: x[2])
        
        x1_min, x2_min = min_pt[1]
        f_min = min_pt[2]
        
        print(f"\nРезультат (минимум):")
        print(f"   Точка экстремума: ({x1_min}, {x2_min})")
        print(f"   Значение функции: f_min = {f_min}")
        print(f"   Численное значение: f_min ≈ {f_min.evalf()}")
        
        x1_max, x2_max = max_pt[1]
        f_max = max_pt[2]
        
        print(f"\nРезультат (максимум):")
        print(f"   Точка экстремума: ({x1_max}, {x2_max})")
        print(f"   Значение функции: f_max = {f_max}")
        print(f"   Численное значение: f_max ≈ {f_max.evalf()}")
        
        return {
            'min': {'x1': x1_min, 'x2': x2_min, 'f_value': f_min},
            'max': {'x1': x1_max, 'x2': x2_max, 'f_value': f_max}
        }
    
    return None


if __name__ == "__main__":
    x1, x2 = symbols('x1 x2')
    
    f = x1**2 - x2**2
    phi = x1 + 4*x2 - 3
    
    result = find_global_extremum(f, phi)