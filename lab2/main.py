import numpy as np
import math
from solver import execute

if __name__ == "__main__":
    def expr(x):
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * x**2)) - np.exp(0.5 * np.cos(2 * np.pi * x)) + np.e + 20

    l_border = -5
    r_border = 5

    eps = 0.01

    results = execute(
        expr=expr,
        start=l_border,
        end=r_border,
        tol=eps,
    )

    print(f"Приближённый минимум найден в точке x ≈ {results['x_min']:.6f}")
    print(f"Значение функции в минимуме f(x) ≈ {results['f_min']:.6f}")
    print(f"Количество итераций: {results['iterations']}")
    print(f"Затраченное время: {results['time_sec']:.4f} секунд")
