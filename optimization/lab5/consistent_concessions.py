import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

print("Метод последовательных уступок")
criteria = [
    {'a': 1.0, 'b': 2.0, 'c': 1.0, 'd': 3.0},
    {'a': 1.0, 'b': -3.0, 'c': 1.0, 'd': -1.0}, # выбираем сами
    {'a': 1.0, 'b': 4.0, 'c': 1.0, 'd': -2.0},
]

def f(x, i):
    return criteria[i]['a'] * (x[0] - criteria[i]['b'])**2 + \
           criteria[i]['c'] * (x[1] - criteria[i]['d'])**2

mins = [minimize(lambda x: f(x, i), [criteria[i]['b'], criteria[i]['d']], 
                method = 'BFGS').x for i in range(3)]

f1_star, f2_star = f(mins[0], 0), f(mins[1], 1)

# начинаем метод
pareto_points = []
for delta1 in np.linspace(0.5, 10.0, 10):
    for delta2 in np.linspace(0.5, 15.0, 10):
        constraints = [
            {'type': 'ineq', 'fun': lambda x: delta1 - (f(x, 0) - f1_star)},
            {'type': 'ineq', 'fun': lambda x: delta2 - (f(x, 1) - f2_star)}
        ]
        result = minimize(lambda x: f(x, 2), mins[2], method = 'SLSQP', constraints = constraints)
        if result.success:
            pareto_points.append(result.x)
        if len(pareto_points) >= 10:
            break
    if len(pareto_points) >= 10:
        break

pareto_points = np.array(pareto_points[:10])
pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

plt.figure(figsize = (10, 8))
colors = ['blue', 'orange', 'green']
for i in range(3):
    plt.plot(mins[i][0], mins[i][1], 'o', color = colors[i], markersize = 12, label = f'Минимум f{i+1}')

plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'r-o', linewidth = 2, markersize = 8, label = 'Множество Парето')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Множество Парето')
plt.grid(True, alpha = 0.3)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()

print("Точки Парето:")
for i, pt in enumerate(pareto_points):
    print(f"{i + 1} точка: ({pt[0]:.4f}, {pt[1]:.4f})")