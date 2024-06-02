import numpy as np
import matplotlib.pyplot as plt

# Data yang diberikan
x_values = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y_values = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Fungsi untuk interpolasi Lagrange
def lagrange_interpolation(x, x_points, y_points):
    def L(k, x):
        terms = [(x - x_points[j]) / (x_points[k] - x_points[j]) for j in range(len(x_points)) if j != k]
        return np.prod(terms)
    
    result = sum(y_points[k] * L(k, x) for k in range(len(x_points)))
    return result

# Fungsi untuk interpolasi Newton
def newton_interpolation(x, x_points, y_points):
    def divided_diff(x_points, y_points):
        n = len(y_points)
        coef = np.zeros([n, n])
        coef[:, 0] = y_points

        for j in range(1, n):
            for i in range(n - j):
                coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x_points[i + j] - x_points[i])

        return coef[0, :]

    coeffs = divided_diff(x_points, y_points)
    n = len(coeffs)
    polynomial = coeffs[-1]

    for k in range(1, n):
        polynomial = polynomial * (x - x_points[-k - 1]) + coeffs[-k - 1]

    return polynomial

# Plotting
x_plot = np.linspace(5, 40, 400)
y_lagrange = [lagrange_interpolation(x, x_values, y_values) for x in x_plot]
y_newton = [newton_interpolation(x, x_values, y_values) for x in x_plot]

plt.figure(figsize=(12, 6))
plt.plot(x_plot, y_lagrange, label='Lagrange Interpolation', color='blue')
plt.plot(x_plot, y_newton, label='Newton Interpolation', color='green')
plt.scatter(x_values, y_values, color='red', zorder=5)
plt.title('Interpolation using Lagrange and Newton Methods')
plt.xlabel('Tegangan, x (kg/mm²)')
plt.ylabel('Waktu patah, y (jam)')
plt.legend()
plt.grid(True)
plt.show()
