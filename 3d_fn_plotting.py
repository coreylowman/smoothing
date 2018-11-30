import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from functions import YangFlockton, YaoLiuLin, get_bounds
from smoothing import SphereSmoother, GaussianSmoother

f = YangFlockton.f2

mins, maxs = get_bounds(f)
zmin, zmax = 0, 25

# Make data.
x = np.linspace(mins[0], maxs[0], 10 * (maxs[0] - mins[0]))
y = np.linspace(mins[0], maxs[0], 10 * (maxs[0] - mins[0]))
X, Y = np.meshgrid(x, y)

smoother = SphereSmoother(f, radius=2.0, num_points=20, num_generations=80, reduce_pct=0.7, on=True)
# smoother = GaussianSmoother(f, stddev=3.0, num_points=10, num_generations=80, reduce_pct=0.7, on=True)

Z = X.copy()
SZ = X.copy()
for i in range(len(X)):
    for j in range(len(X[i])):
        Z[i, j] = f((X[i, j], Y[i, j]))
        SZ[i, j] = smoother.fn((X[i, j], Y[i, j]))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm')
ax.set_zlim((zmin, zmax))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, SZ, cmap='coolwarm')
ax.set_zlim((zmin, zmax))

plt.show()
