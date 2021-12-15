from math import nan
import rootFunctions as rootF
import linearSystems as linearS
import baseConversion as baseC
import numpy as np

import sys

def f(x):
    return np.log(abs(x)) - 2*np.sin(2*x)

def g(x):
    return np.log(abs(2*x)) + 2*np.cos(2*x)

def main():
    ##print(rootF.bisection_method(f, interval_inf=1, interval_sup=2, epsilon_x=1e-1, epsilon_y=1e-1, info=True))
    ##print(rootF.false_position_method(f, interval_inf=1, interval_sup=2, epsilon_x=1e-1, epsilon_y=1e-1, info=True))
    #print(rootF.secant_method(f, previous_x=1.5, previous_x2=1.7, iter_max = 3, info=True))
    #dimension = input()
    
    
    #matrix_a, matrix_b = linearS.input_matrix(dimension, True)
    
    #print(matrix_a)
    #print()
    #print(matrix_b)
    print()
    #linearS.gauss_jacobi(matrix_a,matrix_b,f , iter_max=10, info=True)
    #matrix_l, matrix_u = linearS.lu_factorization(matrix_a, matrix_b)
    #print(matrix_l)
    #print()
    #https://www.youtube.com/watch?v=87pkkBm2exw

    #questao 1
    


def questao1():
    print("0.35 em binario eh", float(baseC.float_to_fbinary("0.35",11)))
    x_approx = float(baseC.binary_to_float("0.0101100110"))
    print(baseC.binary_to_float("0.0101100110"))
    d = 10000 * abs(0.35- x_approx)
    print(d)

def questao2():
    #1 - false
    #2 - false (teste do bolzano)
    print("nenhuma")

def questao3():
    #1  (i) sob o ponto de vista numérico, a codificação A(A - B) não é equivalente a codificação A^2 - AB;
    True
    #2  (ii) numericamente, quando uma operação resulta em INF é um indicativo de ocorreu uma divisão de um número por zero. Por outro lado, uma operação que resulte em NAN ocorre quando existe uma multiplicação de 0 por INF,
    # ou uma divisão de INF por INF, por exemplo. São verdadeiras as afirmações:
    print(122/sys.float_info.min)
    print(0*float('inf'))
    print(float('inf')/float('inf'))
    True

def questao4():
    dimension = 3
    S= linearS.input_matrix(dimension)
    R= linearS.input_matrix(dimension)
    first_norm = linearS.first_norm(S)
    
    inf_norm = linearS.first_norm(S.transpose())
    print(S)
    print(first_norm, inf_norm, first_norm>inf_norm)
    print()
    print(R)
    print()
    Q = linearS.cholesky(R)
    print(Q) ##Nao eh positiva definida

def questao5():
    A = 1.3
    B = 1.5
    biss = rootF.bisection_method(f, A, B,iter_max=2, info=True)
    sec = rootF.secant_method(f, previous_x=B, previous_x2=A,iter_max=2, info=True)

    D = 100 * abs(biss-sec)
    print(D, '[2,4]')

def questao5Lucas():
    A = 0.7
    B = 1.0
    biss = rootF.false_position_method(g, A, B,iter_max=2, info=True)
    sec = rootF.secant_method(g, previous_x=B, previous_x2=A,iter_max=3, info=True)

    D = 10**7* abs(biss-sec)
    print(D)


def questao6():
    dimension = 3
    matrix_a, matrix_b = linearS.input_matrix(dimension, True)
    L, U = linearS.lu_factorization(matrix_a, matrix_b)
    print(matrix_a)
    print()
    print(matrix_b)
    print()
    print()
    print(L)
    print()
    print(U)

    soma = 0
    for i in range(dimension):
        for j in range(dimension):
            soma += abs(L[i,j]) + abs(U[i,j])
    print(soma, "[24,26]")



if __name__ == '__main__':
    questao5Lucas()

'''
Entradas:

5
2
2
-2
4
-3
2
-2
6
9
-1
6


S=
4
2
4
3
1
2
-3
2
1

R=
4
2
1
2
1
1
1
1
1

SR:
4
2
4
3
1
2
-3
2
1
4
2
1
2
1
1
1
1
1


'''