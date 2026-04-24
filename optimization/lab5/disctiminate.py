import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

criteria = [
    {'a': 1.0, 'b': 2.0, 'c': 1.0, 'd': 3.0},
    {'a': 1.0, 'b': -3.0, 'c': 1.0, 'd': -1.0},
    {'a': 1.0, 'b': 4.0, 'c': 1.0, 'd': -2.0},
]

def f(x, i):
    return criteria[i]['a'] * (x[0] - criteria[i]['b'])**2 + \
           criteria[i]['c'] * (x[1] - criteria[i]['d'])**2

mins = [minimize(lambda x: f(x, i), [criteria[i]['b'], criteria[i]['d']], 
                method='BFGS').x for i in range(3)]

print("\nМинимумы критериев:")
print(f"  f₁* = {f(mins[0], 0):.2f} в точке ({mins[0][0]:.1f}, {mins[0][1]:.1f})")
print(f"  f₂* = {f(mins[1], 1):.2f} в точке ({mins[1][0]:.1f}, {mins[1][1]:.1f})")
print(f"  f₃* = {f(mins[2], 2):.2f} в точке ({mins[2][0]:.1f}, {mins[2][1]:.1f})")


print("Оптимизируем: F = α₁·f₁ + α₂·f₂ → min")
print("Ограничение: f₃ ≤ limit")
print()

pareto_points = []
pareto_f1 = []
pareto_f2 = []
pareto_f3 = []
pareto_weights = []
pareto_limits = []

f3_max = f(np.array([-10, -10]), 2)

for alpha1 in np.linspace(0, 1, 10):
    for alpha2 in np.linspace(0, 1, 10):
        if alpha1 + alpha2 > 1:
            continue
        
        for limit_factor in np.linspace(0.1, 1.0, 10):
            limit_f3 = f(mins[2], 2) + limit_factor * (f3_max - f(mins[2], 2))
            
            def F(x):
                return alpha1 * f(x, 0) + alpha2 * f(x, 1)
            
            constraints = [{'type': 'ineq', 'fun': lambda x: limit_f3 - f(x, 2)}]
            
            x0 = np.mean(mins[:2], axis=0)
            
            result = minimize(F, x0, method='SLSQP', constraints=constraints)
            
            if result.success:
                pareto_points.append(result.x)
                pareto_f1.append(f(result.x, 0))
                pareto_f2.append(f(result.x, 1))
                pareto_f3.append(f(result.x, 2))
                pareto_weights.append((alpha1, alpha2))
                pareto_limits.append(limit_f3)

pareto_points = np.array(pareto_points)
pareto_f1 = np.array(pareto_f1)
pareto_f2 = np.array(pareto_f2)
pareto_f3 = np.array(pareto_f3)

unique_idx = []
for i in range(len(pareto_points)):
    is_unique = True
    for j in unique_idx:
        if np.linalg.norm(pareto_points[i] - pareto_points[j]) < 1e-6:
            is_unique = False
            break
    if is_unique:
        unique_idx.append(i)
    if len(unique_idx) >= 10:
        break

pareto_points = pareto_points[unique_idx]
pareto_f1 = pareto_f1[unique_idx]
pareto_f2 = pareto_f2[unique_idx]
pareto_f3 = pareto_f3[unique_idx]

idx = np.argsort(pareto_points[:, 0])
pareto_points = pareto_points[idx]
pareto_f1 = pareto_f1[idx]
pareto_f2 = pareto_f2[idx]
pareto_f3 = pareto_f3[idx]

print(f"\nНайдено {len(pareto_points)} точек Парето")
print("\nТочки Парето:")
print("-" * 70)
print(f"{'№':<4} | {'x₁':<10} | {'x₂':<10} | {'f₁':<10} | {'f₂':<10} | {'f₃':<10}")
print("-" * 70)

for i in range(len(pareto_points)):
    print(f"{i+1:<4} | {pareto_points[i][0]:<10.4f} | {pareto_points[i][1]:<10.4f} | "
          f"{pareto_f1[i]:<10.4f} | {pareto_f2[i]:<10.4f} | {pareto_f3[i]:<10.4f}")

plt.figure(figsize=(10, 8))

colors = ['blue', 'orange', 'green']
for i in range(3):
    plt.plot(mins[i][0], mins[i][1], 'o', color=colors[i], markersize=12, 
             label=f'Минимум f{i+1}')

plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'r-o', linewidth=2, 
         markersize=8, label='Множество Парето')

plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Дискриминационный метод: множество Парето в пространстве переменных')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()