import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as smp
import linearSystems as linearS
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy.linalg import solve


#Método básico de interpolação pela matriz de Vandermonde
def vandermonde_interpolation(points, simplify=False, info=False):
    x = smp.Symbol('x')
    coefficients = []
    
    matrixA = np.zeros((len(points),len(points)))
    matrixB = np.zeros(len(points))
    for i in range(len(points)):
        for j in range(len(points)):
            matrixA[i,j] = points[i][0]**j
        matrixB[i] = points[i][1]
    
    ma, mb, mx = linearS.gaussian_elimination(matrixA, matrixB, False)
    
    summ=0
    for i in range(len(mx)):
        summ += mx[i]*x**i

    if simplify:
        summ = smp.simplify(summ)

    if info:
        print('Interpolação pela matriz de Vandermonde:')
        print('p(x) =', summ)
    return summ

#Método de ajuste linear por mínimos quadrados
def least_squares_method(points, functions, simplify=False, info=False):
    x = smp.Symbol('x')
    DIMENSION = len(functions)
    A = np.zeros([DIMENSION, DIMENSION])
    B = np.zeros(DIMENSION)
    f = []
    g_arr = []

    for point in points:
        f.append(point[1])
    
    for function in functions:
        g = []

        for point in points:
            g.append(function(point[0]))

        g_arr.append(g)

    for i in range(DIMENSION):
        for j in range(DIMENSION):
            A[i,j] = scalar_product(g_arr[i], g_arr[j])
        B[i] = scalar_product(g_arr[i], f)
    
    coefficients = solve(A, B)
    summ = 0
    for i in range(len(functions)):
        summ += coefficients[i]*functions[i](x)

    if simplify:
        summ = smp.simplify(summ)

    if info:
        print('Ajuste linear por MMQ:')
        print('phi(x) = ', summ)
    return summ 

#Produto escalar
def scalar_product(vector1, vector2):
    summ = 0

    for i in range(len(vector1)): 
        summ += vector1[i] * vector2[i]

    return summ

'''
def test_approx_function(x, coefficients, functions):
    summ = 0

    for i in range(len(coefficients)):
        summ += coefficients[i] * functions[i](x)
    
    return summ

'''
#Erro quadrático
def quadratic_error(points, function, info=False):
    summ = 0
    for point in points:
        summ += (point[1]- function(point[0]))**2
    
    if info:
        print('O erro quadrático é:', summ)

    return summ


def r_squared(points, function):
    sum1 = quadratic_error(points, function)
    fbar = 0
    for point in points:
        fbar += point[1]
    fbar /= len(points)

    sum2 = 0
    for point in points:
        sum2 += (fbar- function(point[0]))**2

    return 1 - sum1/(sum2 + sum1)

'''
def lagrangian_method_lop(points, x, index): ##resolver depois com sympy
    lagrangian_functions_values = []
    summ = 0
    for i in range(len(points)):
        product = 1
        for j in range(len(points)):
            if i != j:
                product *= (x - points[j][0])/(points[i][0]-points[j][0])
        lagrangian_functions_values.append(product)

        #show_graph(func,-5,5, [])

        summ += points[i][1]*lagrangian_functions_values[i]
        #print(f"{+points[i][1]} * {lagrangian_functions[i](x)}")

    return summ, lagrangian_functions_values[index]
'''
#Método de interpolação de Lagrange
def lagrangian_method_smp(points, simplify=False, info=False, coefficients=False, value=0): 
    x = smp.Symbol('x')
    summ = 0
    #products=0
    for i in range(len(points)):
        product = 1
        for j in range(len(points)):
            if i != j:
                product *= (x - points[j][0])/(points[i][0]-points[j][0])
        
        if coefficients:
            print(i, smp.expand(smp.simplify(product)),'---', evaluate_sympy(product, value))
            #products += product
        summ += points[i][1]*product

    #if coefficients:
    #    print('product sum', smp.expand(smp.simplify(products)))
    if simplify:
        summ = smp.simplify(summ)

    if info:
        print('Interpolação pelo método de Lagrange:')
        print('p(x) =', summ)

    return summ

#Avalia uma expressão para um dado x
def evaluate_sympy(expression, value, info=False):
    fun = lambdify_sympy(expression);
    evaluation = fun(value)
    if info:
        print('O valor avaliado na expressão é:', evaluation)
    return evaluation

def lambdify_sympy(expression):
    x = smp.Symbol('x')
    return smp.lambdify(x, expression, 'numpy')
    
#Operador de diferença dividida simples
def divided_dif_operator_simple(points):
    difs_ord=[]
    for i in range(len(points)):
        list_dif = []
        if i==0:
            for j in range(len(points)):
                list_dif.append(points[j][1])
        else:
            for j in range(len(difs_ord[i-1])-1):
                list_dif.append(difs_ord[i-1][j+1] - difs_ord[i-1][j])
        difs_ord.append(list_dif)

    return difs_ord

#Método de Newton de interpolação
def newton_interpolation(points, simplify=False, info=False):
    x = smp.Symbol('x')
    summ = 0
    for i in range(len(points)):
        product = 1
        for j in range(i):
            product*=(x-points[j][0])
        summ += product*divided_dif_operator(points[0:i+1])
        if info:
            print(divided_dif_operator(points[0:i+1]))

    if simplify:
        summ = smp.simplify(summ)

    if info:
        print('p(x) = ', end='')
        print(summ)
    
    return summ

#Operador de diferença dividida
def divided_dif_operator(points):
    if len(points)==0:
        return 0
    elif len(points)==1:
        return points[0][1]
    else:
        return (divided_dif_operator(points[1:])-divided_dif_operator(points[:len(points)-1]))/(points[len(points)-1][0]-points[0][0])

#Método de Gregory-newton de interpolação
def gregory_newton_interpolation(points, simplify=False, info=False):
    x = smp.Symbol('x')
    summ = 0
    h = points[1][0] - points[0][0]
    z =(x - points[0][0])/h
    difs = divided_dif_operator_simple(points)

    for i in range(len(points)):
        product = 1
        for j in range(i):
            product*=(z-j)
        summ += product*difs[i][0]/math.factorial(i)

    if simplify:
        summ = smp.simplify(summ)

    if info:
        print('p(x) = ', end='')
        print(summ)

    return summ

#Método de Simpson para estimar integrais
def simpsons_method(points, info):
    DIMENSAO = len(points)
    H = points[1][0] - points[0][0]
    summ = 0
    coef = 1
    for i in range(DIMENSAO):
        if i == 0 or i == DIMENSAO-1:
            coef = 1
        elif i%2==0:
            coef = 2
        else:
            coef = 4
        summ += coef*points[i][1]

    summ *= H/3
    if info:
        print('Valor aproximado da integral pelo método de Simpson:', summ)

    return summ

#Método de Simpson 3/8 para estimar integrais
def simpsons_method_3_8(points, info):
    DIMENSAO = len(points)
    H = points[1][0] - points[0][0]
    summ = 0
    coef = 1
    for i in range(DIMENSAO):
        if i == 0 or i == DIMENSAO-1:
            coef = 1
        elif i%3==0:
            coef = 2
        else:
            coef = 3
        summ += coef*points[i][1]

    summ *= 3*H/8
    if info:
        print('Valor aproximado da integral pelo método de Simpson 3/8:', summ)

    return summ

#Método dos trapézios para estimar integrais
def trapezoid_method(points, info=False):
    DIMENSAO = len(points)
    H = points[1][0] - points[0][0]
    summ = 0
    coef = 1
    for i in range(DIMENSAO):
        if i == 0 or i == DIMENSAO-1:
            coef = 1
        else:
            coef = 2
        summ += coef*points[i][1]

    summ *= H/2
    if info:
        print('Valor aproximado da integral pelo método dos trapézios:', summ)

    return summ

def cubic_spline(points):
    x = []
    y = []
    for point in points:
        x.append(points[0])
        y.append(points[1])

    return InterpolatedUnivariateSpline(x,y)


'''
def trapezoid_method_new(points):
    DIMENSAO = len(points)
    summ = 0
    for i in range(1,DIMENSAO-1):
        summ += points[i][1]

    return (summ*2 +(points[0][1] + points[DIMENSAO-1][1])) * (points[1][0] - points[0][0])/2
#def simpsons_error(points):
#    error = 0
#''
'''
#Gerar pontos (x, f(x)) dada uma função f, intervalo e quantidade de pontos
def generate_points(function, interval_inf, interval_sup, num_points):
    x = np.linspace(interval_inf, interval_sup, num=num_points)
    points = []
    for x_i in x:
        points.append((x_i, function(x_i)))
    
    return points

#Método de Euler para EDO de grau 1
def euler_method_edo(function, wanted_x, initial_conditions,  steps):
    x, y = initial_conditions
    H = (wanted_x - x)/steps

    for i in range(steps):
        y = y + H*function(x, y)
        x += H
    
    return y    

#Método de Euler modificado para EDO de grau 1
def modified_euler_method_edo(function, wanted_x, initial_conditions,  steps):
    x, y = initial_conditions
    H = (wanted_x - x)/steps

    for i in range(steps):
        y = y + H*(function(x, y) + function(x + H, y + H*function(x, y)))/2
        x += H
    
    return y
    
#Método de Runge-Kutta de ordem 4 para EDO
def runge_kutta_method_edo(function, wanted_x, initial_conditions,  steps):
    x, y = initial_conditions
    H = (wanted_x - x)/steps

    for i in range(steps):
        K1 = function(x, y)
        K2 = function(x + H/2, y + K1*H/2)
        K3 = function(x + H/2, y + K2*H/2)
        K4 = function(x + H, y + K3*H)
        y = y + H*(K1 + 2*K2 + 2*K3 + K4)/6
        x += H
    
    return y 

#Método de Runge-Kutta de ordem 4 para EDO de 2a ordem
def runge_kutta_method_edo_2(function, wanted_x, initial_conditions,  steps):
    x, y1, y2 = initial_conditions
    H = (wanted_x - x)/steps

    for i in range(steps):
        K1 = function(x, y)
        K2 = function(x + H/2, y + K1*H/2)
        K3 = function(x + H/2, y + K2*H/2)
        K4 = function(x + H, y + K3*H)
        y = y + H*(K1 + 2*K2 + 2*K3 + K4)/6
        x += H
    
    return y    

#Mostra o gráfico das funções
def show_graph(function, interval_start, interval_finish, points=[]):
    x = np.linspace(interval_start, interval_finish)
    plt.plot(x, function(x), color="purple")
    for point in points:
        plt.plot([point[0]],[point[1]], marker="o", markersize=1, markeredgecolor="red")
    plt.grid()
    plt.show()
