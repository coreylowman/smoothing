import matplotlib.pyplot as plt
import numpy as np
from functions import YangFlockton, YaoLiuLin, get_bounds
from smoothing import SphereSmoother, GaussianSmoother

f = YaoLiuLin.f9

mins, maxs = get_bounds(f)
ymin, ymax = 0, 45

# Make data.
X = np.linspace(mins[0], maxs[0], 10 * (maxs[0] - mins[0]))

smoother = SphereSmoother(f, radius=0.5, num_points=30, num_generations=80, reduce_pct=0.7, on=True)
# smoother = GaussianSmoother(f, stddev=3.0, num_points=10, num_generations=80, reduce_pct=0.7, on=True)

Z = X.copy()
SZ = X.copy()
for i in range(len(X)):
    Z[i] = f((X[i],))
    SZ[i] = smoother.fn((X[i],))

plt.figure()
plt.plot(X, Z)
plt.ylim((ymin, ymax))

plt.figure()
plt.plot(X, SZ)
plt.ylim((ymin, ymax))

plt.show()
