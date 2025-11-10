import numpy as np
import math
import matplotlib.pyplot as plt
import time


def global_minimize(f, start, end, tol=1e-2, max_steps=20000, L=None, L_multiplier=1.1, init_samples=3, track_history=True):
    t_start = time.time()

    X = np.linspace(start, end, init_samples)
    Y = np.array([float(f(x)) for x in X])

    def estimate_L(x_vals, y_vals):
        delta = np.diff(y_vals) / np.maximum(np.diff(x_vals), 1e-15)
        return max(1.0, np.max(np.abs(delta)) if delta.size else 1.0)

    def sort_xy(x_vals, y_vals):

        idx = np.argsort(x_vals)
        return x_vals[idx], y_vals[idx]

    def compute_interval_extrema(x_vals, y_vals, L_val):
        dx = np.diff(x_vals)
        dy = np.diff(y_vals)
        x_candidate = 0.5*(x_vals[:-1] + x_vals[1:]) - dy / (2 * L_val)
        R = 0.5*(y_vals[:-1] + y_vals[1:] - L_val * dx)
        return x_candidate, R

    def global_lower_bound(x_vals, y_vals, L_val):
        _, R = compute_interval_extrema(x_vals, y_vals, L_val)
        return np.min(R)

    L = estimate_L(X, Y) * L_multiplier if L is None else float(L)
    X, Y = sort_xy(X, Y)
    history = []
    step = 0

    while step < max_steps:
        step += 1

        L = max(L, estimate_L(X, Y) * L_multiplier)

        X_candidates, R_values = compute_interval_extrema(X, Y, L)
        idx_new = int(np.argmin(R_values))
        x_new = float(X_candidates[idx_new])
        y_new = float(f(x_new))

        X = np.append(X, x_new)
        Y = np.append(Y, y_new)
        X, Y = sort_xy(X, Y)

        best_idx = int(np.argmin(Y))
        x_best, y_best = float(X[best_idx]), float(Y[best_idx])
        glb = float(global_lower_bound(X, Y, L))

        if track_history:
            history.append((step, x_new, y_new, L, x_best, y_best, glb))

        if y_best - glb <= tol:
            break

    X_candidates, R_values = compute_interval_extrema(X, Y, L)
    elapsed_time = time.time() - t_start

    return {
        "x_min": x_best,
        "f_min": y_best,
        "iterations": step,
        "time_sec": elapsed_time,
        "L_used": float(L),
        "X_samples": X,
        "Y_samples": Y,
        "X_candidates": X_candidates,
        "R_values": R_values,
        "history": history,
        "a": float(start),
        "b": float(end),
        "tol": tol
    }


def plot_results(f, res, base_name="figure"):
    Xs = res["X_samples"]
    Ys = res["Y_samples"]
    Xc = res["X_candidates"]
    R = res["R_values"]
    x_best = res["x_min"]
    y_best = res["f_min"]
    L = res["L_used"]
    a, b = res["a"], res["b"]

    grid = np.linspace(a, b, 2000)
    f_grid = f(grid)

    plt.figure(figsize=(10,6), dpi=140)

    plt.plot(grid, f_grid, color='royalblue', linewidth=2, label="f(x)")

    for i in range(len(Xs)-1):
        seg = np.linspace(Xs[i], Xs[i+1], 60)
        lower = np.maximum(
            Ys[i] - L*np.abs(seg - Xs[i]),
            Ys[i+1] - L*np.abs(Xs[i+1] - seg)
        )
        plt.plot(seg, lower, linestyle='--', color='darkorange', linewidth=1.2)

    plt.scatter(Xs, Ys, s=30, color='green', edgecolor='black', label="samples")

    if len(Xc) > 0:
        plt.scatter(Xc, R, s=50, marker="^", color='purple', label="candidates")

    plt.axhline(y_best, linestyle=":", color='red', linewidth=1.5, label="текущее min")
    plt.axvline(x_best, linestyle=":", color='red', linewidth=1.5)
    plt.scatter([x_best], [y_best], s=60, marker="X", color='red', label="min точка")

    plt.title("Глобальная оптимизация функции (Липшиц)", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    png_file = f"{base_name}.png"
    pdf_file = f"{base_name}.pdf"
    plt.tight_layout()
    plt.savefig(png_file)
    plt.savefig(pdf_file)
    plt.show()
    plt.close()

    return png_file, pdf_file


def execute(expr, start, end, tol=1e-2):
    res = global_minimize(expr, start, end, tol=tol)
    plot_results(expr, res, 'data/output')
    return res
