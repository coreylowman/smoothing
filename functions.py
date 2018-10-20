from math import sin, cos, pi, e, exp, sqrt, floor
from functools import reduce
import operator
import random


def squared(x_i):
    return x_i ** 2


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


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
            return i * (x[i] ** 4) + random.random()

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
        sum2 = sum(map(lambda x_i: cos(2 * pi * x_i), x))
        return 20 + e - 20 * exp(-0.2 * sqrt(sum1 / n)) - exp(sum2 / n)

    @staticmethod
    def f11(x):
        """n = 30, x_i \in [-600, 600], f_min = 0"""
        return 0

    @staticmethod
    def f12(x):
        """n = 30, x_i \in [-50, 50], f_min = 0"""
        return 0

    @staticmethod
    def f13(x):
        """n = 30, x_i \in [-50, 50], f_min = 0"""
        return 0

    @staticmethod
    def f14(x):
        """n = 2, x_i \in [-65.536, 65.536], f_min = 1"""
        return 0

    @staticmethod
    def f15(x):
        """n = 4, x_i \in [-5, 5], f_min = 0.0003075"""
        return 0

    @staticmethod
    def f16(x):
        """n = 2, x_i \in [-5, 5], f_min = -1.0316285"""
        return 0

    @staticmethod
    def f17(x):
        """n = 2, [-5, 10] x [0, 15], f_min = 0.398"""
        return 0

    @staticmethod
    def f18(x):
        """n = 2, x_i \in [-2, 2], f_min = 3"""
        return 0

    @staticmethod
    def f19(x):
        """n = 4, x_i \in [0, 1], f_min = -3.86"""
        return 0

    @staticmethod
    def f20(x):
        """n = 6, x_i \in [0, 1], f_min = -3.32"""
        return 0

    @staticmethod
    def f21(x):
        """n = 4, x_i \in [0, 1], f_min = -10"""
        return 0

    @staticmethod
    def f22(x):
        """n = 4, x_i \in [0, 10], f_min = -10"""
        return 0

    @staticmethod
    def f23(x):
        """n = 4, x_i \in [0, 10], f_min = -10"""
        return 0
