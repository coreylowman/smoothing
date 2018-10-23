from math import sin, cos, pi, e, exp, sqrt, floor
from functools import reduce
import operator
import random


def squared(x_i):
    return x_i ** 2


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def sin2(x):
    return sin(x) ** 2


class YangFlockton:
    """
    Yang, D., & Flockton, S. J. (1995). Evolutionary algorithms with a coarse-to-fine function
            smoothing. In Evolutionary Computation, 1995., IEEE International Conference on,
            Vol. 2, pp. 657{662. IEEE.
    """

    @staticmethod
    def f1(x):
        """x_i \in [-5.12, 5.12]"""
        n = len(x)
        return n + sum(map(lambda x_i: x_i ** 2 - cos(2 * pi * x_i), x))

    @staticmethod
    def f2(x):
        """x_i \in [-30, 30]"""
        n = len(x)
        sum1 = sum(map(squared, x))
        sum2 = sum(map(lambda x_i: cos(2 * pi * x_i), x))
        return 20 + e - 20 * exp(-0.2 * sqrt(sum1 / n)) - exp(sum2 / n)


class YaoLiuLin:
    """
    Yao, X., Liu, Y., & Lin, G. (1999). Evolutionary programming made faster. IEEE Trans-
        actions on Evolutionary computation, 3 (2), 82{102.
    """

    @staticmethod
    def f1(x):
        """n = 30, x_i \in [-100, 100], f_min = 0"""
        return sum(map(squared, x))

    @staticmethod
    def f2(x):
        """n = 30, x_i \in [-10, 10], f_min = 0"""
        return sum(map(abs, x)) + prod(map(abs, x))

    @staticmethod
    def f3(x):
        """n = 30, x_i \in [-100, 100], f_min = 0"""
        return sum(map(lambda i: sum(x[:i]) ** 2, range(len(x))))

    @staticmethod
    def f4(x):
        """n = 30, x_i \in [-100, 100], f_min = 0"""
        return max(map(abs, x))

    @staticmethod
    def f5(x):
        """n = 30, x_i \in [-30, 30], f_min = 0"""

        def inner_f(i):
            return 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2

        return sum(map(inner_f, range(len(x) - 1)))

    @staticmethod
    def f6(x):
        """n = 30, x_i \in [-100, 100], f_min = 0"""
        return sum(map(lambda x_i: floor(x_i + 0.5) ** 2, x))

    @staticmethod
    def f7(x):
        """n = 30, x_i \in [-1.28, 1.28], f_min = 0"""

        def inner_f(i):
            return (i + 1) * (x[i] ** 4)

        return sum(map(inner_f, range(len(x))))

    @staticmethod
    def f8(x):
        """n = 30, x_i \in [-500, 500], f_min = -12569.5"""
        return sum(map(lambda x_i: -x_i * sin(sqrt(abs(x_i))), x))

    @staticmethod
    def f9(x):
        """n = 30, x_i \in [-5.12, 5.12], f_min = 0"""
        return sum(map(lambda x_i: x_i ** 2 - 10 * cos(2 * pi * x_i) + 10, x))

    @staticmethod
    def f10(x):
        """n = 30, x_i \in [-32, 32], f_min = 0"""
        n = len(x)
        sum1 = sum(map(squared, x))
        term1 = -20 * exp(-0.2 * sqrt(sum1 / n))

        sum2 = sum(map(lambda x_i: cos(2 * pi * x_i), x))
        term2 = -exp(sum2 / n)
        return 20 + e + term1 + term2

    @staticmethod
    def f11(x):
        """n = 30, x_i \in [-600, 600], f_min = 0"""
        term1 = sum(map(squared, x)) / 4000.0
        term2 = prod(map(lambda i: cos(x[i] / sqrt(i + 1)), x))
        return 1 + term1 - term2

    @staticmethod
    def f12(x):
        """n = 30, x_i \in [-50, 50], f_min = 0"""

        def u(x_i, a, k, m):
            if x_i > a:
                return k * ((x_i - a) ** m)
            elif x_i < -a:
                return k * ((-x_i - a) ** m)
            else:
                return 0

        n = len(x)
        y = type(x)(map(lambda x_i: 1 + 0.25 * (x_i + 1), x))

        term1 = 10 * sin2(pi * y[0])
        term2 = sum(map(lambda i: ((y[i] - 1) ** 2) * (1 + 10 * sin2(pi * y[i + 1])), range(n - 1)))
        term3 = (y[-1] - 1) ** 2

        term4 = sum(map(lambda x_i: u(x_i, 10, 100, 4), x))

        return (pi / n) * (term1 + term2 + term3) + term4

    @staticmethod
    def f13(x):
        """n = 30, x_i \in [-50, 50], f_min = 0"""

        def u(x_i, a, k, m):
            if x_i > a:
                return k * ((x_i - a) ** m)
            elif x_i < -a:
                return k * ((-x_i - a) ** m)
            else:
                return 0

        n = len(x)

        term1 = sin2(3 * pi * x[0])
        term2 = sum(map(lambda i: ((x[i] - 1) ** 2) * (1 + sin2(3 * pi * x[i + 1])), range(n - 1)))
        term3 = ((x[-1] - 1) ** 2) * (1 + sin2(2 * pi * x[-1]))

        return 0.1 * (term1 + term2 + term3) + sum(map(lambda x_i: u(x_i, 5, 100, 4), x))

    @staticmethod
    def f14(x):
        """n = 2, x_i \in [-65.536, 65.536], f_min = 1"""
        a = [
            [-32, -16, 0, 16, 32] * 5,
            [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5,
        ]

        def u(j):
            term1 = (j + 1)
            term2 = sum(map(lambda i: (x[i] - a[i][j]) ** 6, range(len(x))))
            return 1 / (term1 + term2)

        term1 = (1 / 500)
        term2 = sum(map(u, range(25)))

        return 1 / (term1 + term2)

    @staticmethod
    def f15(x):
        """n = 4, x_i \in [-5, 5], f_min = 0.0003075"""
        a = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
        b = list(map(lambda b_i: 1 / b_i, [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]))

        top = [x[0] * ((b[i] ** 2) + b[i] * x[1]) for i in range(11)]
        bot = [(b[i] ** 2) + b[i] * x[2] + x[3] for i in range(11)]
        q = [top[i] / bot[i] for i in range(11)]

        return sum(map(lambda i: (a[i] - q[i]) ** 2, range(11)))

    @staticmethod
    def f16(x):
        """n = 2, x_i \in [-5, 5], f_min = -1.0316285"""
        return 4 * (x[0] ** 2) - 2.1 * (x[0] ** 4) + (x[0] ** 6) / 3 + x[0] * x[1] - 4 * (x[1] ** 2) + 4 * (x[1] ** 4)

    @staticmethod
    def f17(x):
        """n = 2, [-5, 10] x [0, 15], f_min = 0.398"""
        term1 = (x[1] - 5.1 * (x[0] ** 2) / (4 * (pi ** 2)) + 5 * x[0] / pi - 6) ** 2
        term2 = 10 * (1 - 1 / (8 * pi)) * cos(x[0])
        term3 = 10
        return term1 + term2 + term3

    @staticmethod
    def f18(x):
        """n = 2, x_i \in [-2, 2], f_min = 3"""
        x1, x2 = x
        t1 = 1 + ((x1 + x2 + 1) ** 2) * (19 - 14 * x1 + 3 * (x1 ** 2) - 14 * x2 + 6 * x1 * x2 + 3 * (x2 ** 2))
        t2 = 30 + ((2 * x1 - 3 * x2) ** 2) * (18 - 32 * x1 + 12 * (x1 ** 2) + 48 * x2 - 36 * x1 * x2 + 27 * (x2 ** 2))
        return t1 * t2

    @staticmethod
    def f19(x):
        """n = 4, x_i \in [0, 1], f_min = -3.86"""
        a = [
            [3, 10, 30],
            [0.1, 10, 35],
            [3, 10, 30],
            [0.1, 10, 35]
        ]
        c = [1, 1.2, 3, 3.2]
        p = [
            [0.3689, 0.1170, 0.2673],
            [0.4699, 0.4387, 0.7470],
            [0.1091, 0.8732, 0.5547],
            [0.038150, 0.5743, 0.8828],
        ]

        def v(i, j):
            return -a[i][j] * ((x[j] - p[i][j]) ** 2)

        def u(i):
            return c[i] * exp(sum(map(lambda j: v(i, j), range(3))))

        return -sum(map(u, range(4)))

    @staticmethod
    def f20(x):
        """n = 6, x_i \in [0, 1], f_min = -3.32"""
        a = [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ]
        c = [1, 1.2, 3, 3.2]
        p = [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
        ]

        def v(i, j):
            return -a[i][j] * ((x[j] - p[i][j]) ** 2)

        def u(i):
            return c[i] * exp(sum(map(lambda j: v(i, j), range(6))))

        return -sum(map(u, range(4)))

    @staticmethod
    def f21(x):
        """n = 4, x_i \in [0, 1], f_min = -10"""
        return YaoLiuLin._shekel_function(x, 5)

    @staticmethod
    def f22(x):
        """n = 4, x_i \in [0, 10], f_min = -10"""
        return YaoLiuLin._shekel_function(x, 7)

    @staticmethod
    def f23(x):
        """n = 4, x_i \in [0, 10], f_min = -10"""
        return YaoLiuLin._shekel_function(x, 10)

    @staticmethod
    def _shekel_function(x, m):
        c = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
        a = [
            [4, 4, 4, 4],
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            [6, 6, 6, 6],
            [3, 7, 3, 7],
            [2, 9, 2, 9],
            [5, 5, 3, 3],
            [8, 1, 8, 1],
            [6, 2, 6, 2],
            [7, 3.6, 7, 3.6],
        ]

        def sub(u, v):
            return type(u)(map(lambda i: u[i] - v[i], range(len(u))))

        def dot(u):
            return sum(map(lambda i: u[i] * u[i], range(len(u))))

        return -sum(map(lambda i: 1 / (dot(sub(x, a[i])) + c[i]), range(m)))
