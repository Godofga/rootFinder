import rootFunctions as rootF
import numpy as np
import math


class Polynomial(object):
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def evaluate(self, x):
        return self.A * x**3 + self.B * x**2 + self.C * x + self.D

    def evaluate_dif(self, x):
        return 3 * self.A * x**2 + 2 * self.B * x + self.C


def main():
    A, B, C, D, U, V = float(input()), float(input()), float(input()), float(input()), float(input()), float(input())
    polynomial = Polynomial(A, B, C, D)

    root = rootF.secant_method(polynomial.evaluate, V, U, epsilon_x=1e-4, info=True)



if __name__ == '__main__':
    main()
