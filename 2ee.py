from math import nan
import rootFunctions as rootF
import linearSystems as linearS
import baseConversion as baseC
import adjustment as adjust
import numpy as np
import sympy as smp
from scipy.integrate import simps, trapz

import sys

def f(x):
    '''
    inserir codigo
    '''

def questao1():
    '''
    Os dois ajustamentos lineares são idênticos
    '''

def questao2():
    x =[0, 2, 4, 6, 10]
    y =[1, 12, 16, 24, 31]

    x1 =[0, 2, 4]
    y1 =[1, 12, 16]

    x2 =[ 2, 4, 6, 10]
    y2 =[ 12, 16, 24, 31]


    print('trapezio', trapz(y2,x2))
    print('simpson', simps(y2,x2))
    '''
    Usando a regra Simpson no intervalo t = 2 e t = 10 a distância é de 176.9 metros
    '''

def questao3():
    eq = adjust.newton_interpolation([(1,1),(3,9),(4,16), (5,25)], info=True)
    eq = smp.simplify(eq)
    print(eq)

    '''
    a1 = 9, a2 = 7, a3 = 1
    '''

def questao4():
    value = 2016.5
    points = [(2010,194.9), (2011,196.6), (2012,198.3), (2013, 200), (2014,201.7), (2015, 203.5), (2016,205.2), (2017,206.8), (2018, 208.5),(2019,210.1),(2020,211.8),(2021,213.3),(2022,214.8)]
    points1= [(2013, 200), (2014,201.7), (2015, 203.5), (2016,205.2), (2017,206.8)]
    points2= [(2015, 203.5), (2016,205.2), (2017,206.8)]
    points3= [(2012,198.3), (2013, 200), (2014,201.7), (2015, 203.5), (2016,205.2), (2017,206.8)]
    points4 = [(2016,205.2), (2017,206.8)]
    points5 = [(2014,201.7), (2015, 203.5), (2016,205.2), (2017,206.8)]
    lagr = adjust.lagrangian_method_smp(points3, simplify=True, info=True, coefficients=True, value = value)
    print('Evalutation:')
    adjust.evaluate_sympy(lagr, value, info=True)
    '''
    Interpolando os anos 2012-2017. A população para 2016.5 é -62.31 e a soma das suas funções Lagrangianas L_0 e L_1 é -2.09 não bate
    '''

def questao5():
    '''
    ambas falsas
    (i) A regra trapezoidal não é exata para funções quadráticas
    (ii) A regra de Simpson não é exata para polinômios de quarto grau

    '''

def questao6():
    def f(x,y):
            return -0.042*(y - 22)
    max_temperature = 42
    initial_temperatures = 92
    cool_down_time = 0
    steps = 1
    medicoes = 0

    
    while True:
        current_temperature = adjust.modified_euler_method_edo(f, medicoes * 12, (0,initial_temperatures), max(medicoes,steps))
        if current_temperature < max_temperature:
            break
        medicoes+=1
    
    print(medicoes)

    current_temperature = adjust.modified_euler_method_edo(f, medicoes*12, (0,initial_temperatures), medicoes)
    print(round(current_temperature,4))  
    '''
    3
    38.927
    '''

def questao7():
    points = [(2.2, 7.04), (2.4, 8.16), (3, 12), (3.2, 13.12)]

    print('A quad error - ',adjust.quadratic_error(points, lambda x: 2*x+ np.cos(x)))
    print('L quad error - ',adjust.quadratic_error(points, lambda x: 2*x**2 - np.sqrt(x)))
    print('J quad error - ',adjust.quadratic_error(points, lambda x: np.exp(x) - np.cos(x)))
    print()
    print('A r^2 error - ',adjust.r_squared(points, lambda x: 2*x+ np.cos(x)))
    print('L r^2 error - ',adjust.r_squared(points, lambda x: 2*x**2 - np.sqrt(x)))
    print('J r^2 error - ',adjust.r_squared(points, lambda x: np.exp(x) - np.cos(x)))
    '''
    LJA (pelo r^2)
    '''

def teste():
    #points = [(-1, 0), (0, 1), (1, 5)]

    #functions = [lambda x: x, lambda x: x**2]
    #points = [(0, 2), (smp.pi/2, 1), (smp.pi, 1)]

    #functions = [smp.sin, smp.cos]
    #adjust.vandermonde_interpolation([(-1,5),(2,2), (3,3)])

    #equation = adjust.lagrangian_method_smp([(-1,-5),(1,-1), (2,1), (3,3)], info=True, simplify=True)
    #adjust.evaluate_sympy(equation, 3)

    #print(adjust.divided_dif_operator([(-1,3), (1,4), (2,7)]))

    #expression = adjust.newton_interpolation(points, simplify=True, info=True)
    #adjust.evaluate_sympy(expression, 0, True)

    #expression = adjust.least_squares_method(points, functions, simplify=False, info=True)
    #adjust.evaluate_sympy(expression, 1, True)
    #function = lambda x: np.exp(x**2)
    #points = adjust.generate_points(function, interval_inf=0, interval_sup=1, num_points=11)
    #adjust.trapezoid_method(points, info=True)
    #adjust.simpsons_method(points, info=True)

    #function = lambda x: np.sin(x)**2
    #points = adjust.generate_points(function, interval_inf=0, interval_sup=np.pi, num_points=7)
    #adjust.trapezoid_method(points, info=True)
    #adjust.simpsons_method(points, info=True)
    #adjust.simpsons_method_3_8(points, info=True)
    #points = [(1.1,2.31),(1.4,3.36),(1.5,3.75)]
    #lagr_adj = adjust.lagrangian_method_smp(points,simplify=True, info=True, coefficients=True)
    #lagr_fun = adjust.lambdify_sympy(lagr_adj)
    #print('P',adjust.quadratic_error(points, lambda x: 2.5*x+ np.log(x)))
    #print('Q',adjust.quadratic_error(points, lambda x: 2*x+ np.cos(x)))
    #print('T',adjust.quadratic_error(points, lambda x: x**2 - 2* np.log(x)))
    
    #adjust.least_squares_method(points,functions, simplify=True, info=True)
    #function = lambda x: abs(np.sin(2*x))
    #points = adjust.generate_points(function, interval_inf=2, interval_sup=6, num_points=5)
    #adjust.trapezoid_method(points, info=True)

    
    #print(lagr_fun())
    #adjust.vandermonde_interpolation([(-1,-5),(0,-2),(1,1)],simplify=True, info=True)
    #points = [(0, -32), (2, -18), (4, 28)]
    #greg = adjust.gregory_newton_interpolation(points, simplify=True, info=True)
    points = [(-1,0), (0,0),(1,0)]
    pointsinv = [(0,-1), (0,0),(0,1)]
    functions = [lambda x: x, lambda x:1]

    adjust.least_squares_method(points, functions, simplify=True, info=True)
    adjust.least_squares_method(pointsinv, functions, simplify=True, info=True)
    


    


    '''
    inserir codigo
    '''



if __name__ == '__main__':
    teste()

'''
Entradas:

'''