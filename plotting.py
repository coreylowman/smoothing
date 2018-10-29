import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from functions import YaoLiuLin
from smoothing import SphereSmoother

# Make data.
x = np.linspace(-5, 5, 30)
y = np.linspace(-5, 5, 30)
X, Y = np.meshgrid(x, y)

f = YaoLiuLin.f1
smoother = SphereSmoother(f, 1.0, 0.0, 10)

Z = X.copy()
SZ = X.copy()
for i in range(len(X)):
    for j in range(len(X[i])):
        Z[i, j] = f((X[i, j], Y[i, j]))
        SZ[i, j] = smoother.smoothed_fitness_fn((X[i, j], Y[i, j]))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, SZ, cmap='coolwarm')

plt.show()
