import numpy as np

def barrier_optimization_simple(x0, r, epsilon=0.01):
    def f(x):
        return (x[0] - 2)**2 + (x[1] - 1)**2
    
    def g(x):
        return 1 - 0.25*x[0]**2 - x[1]**2
    
    def h(x):
        return x[0] - 2*x[1] + 1
    
    def barrier_function(x, r, mu_penalty=1000):
        val_g = g(x)
        if val_g <= 1e-12:
            return 1e10
        return f(x) + mu_penalty * h(x)**2 - r * np.log(val_g)
    
    def gradient(f, x, eps=1e-8):
        grad = np.zeros_like(x)
        f0 = f(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (f(x_plus) - f0) / eps
        
        return grad
    
    def gradient_descent(f, x_start, learning_rate=0.01, max_iter=1000, tol=1e-8):
        x = x_start.copy()
        
        for i in range(max_iter):
            grad = gradient(f, x)
            x_new = x - learning_rate * grad
            
            if g(x_new) <= 0:
                learning_rate *= 0.5
                continue
            
            if np.linalg.norm(x_new - x) < tol:
                return x_new, f(x_new), i+1
            
            x = x_new
        
        return x, f(x), max_iter
    
    x_curr = x0.copy()
    r_curr = r
    
    print(f"{'Итер':<5} | {'r':<10} | {'x1':<10} | {'x2':<10} | {'f(x)':<10}")
    print("-"*70)
    
    for k in range(20):
        
        def current_barrier(x):
            return barrier_function(x, r_curr)
        
        x_opt, f_val, inner_iter = gradient_descent(
            current_barrier, 
            x_curr, 
            learning_rate=0.01,
            max_iter=500
        )
        
        print(f"{k+1:<5} | {r_curr:<10.6f} | {x_opt[0]:<10.6f} | {x_opt[1]:<10.6f} | {f_val:<10.6f}")
        
        barrier_val = -r_curr * np.log(g(x_opt))
        if barrier_val < epsilon:
            print("-"*70)
            print(f"Сходимость на итерации {k+1}")
            return x_opt, f_val
        
        r_curr *= 0.2
        x_curr = x_opt.copy()
    
    return x_opt, f_val


if __name__ == "__main__":
    x0 = np.array([-0.1148, 0.4414])
    
    solution, f_min = barrier_optimization_simple(
        x0=x0,
        r=10.0,
        epsilon=0.01
    )
    
    print("РЕЗУЛЬТАТ:")
    print(f"x* = ({solution[0]:.6f}, {solution[1]:.6f})")
    print(f"f* = {f_min:.6f}")