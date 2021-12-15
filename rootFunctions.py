import numpy as np
import matplotlib.pyplot as plt


def bolzanos_test(function, interval_min, interval_max, info=False):
    if info:
        if function(interval_min)*function(interval_max)<0:
            print(f"Existe ao menos uma raiz em [{interval_min},{interval_max}]")
        else:
            print(f"Nada a afirmar sobre [{interval_min},{interval_max}]")
    return function(interval_min)*function(interval_max)<0


def bisection_method(function, interval_inf, interval_sup, epsilon_x=0, epsilon_y=0, iter_max=0, info=False):
    initial_interval = (interval_inf,interval_sup)
    iteration = 0
    half = lambda a, b: (a + b) / 2

    if info:
        print("Bisection method:")
    
    if (epsilon_x == epsilon_y == iter_max == 0) or interval_inf > interval_sup:
        if info:
            print("Wrong parameters")
        return

    if function(interval_inf) == 0:
        return interval_inf
    if function(interval_sup) == 0:
        return interval_sup

    if not bolzanos_test(function, interval_inf, interval_sup):
        if info:
            print("No roots in the interval according to bolzano's (cannot affirm anything)")
        return

    while abs(interval_sup - interval_inf) >= epsilon_x and abs(function(half(interval_inf, interval_sup))) >= epsilon_y and (iter_max == 0 or iteration < iter_max):
        interval_cut = half(interval_inf, interval_sup)

        if info:
            print(f"Iteration = {iteration}; x approx = {interval_cut}; Interval inf = {interval_inf}; Interval sup = {interval_sup}; Interval size = {interval_sup - interval_inf}; f(x) approx = {function(interval_cut)}")
        iteration += 1

        if bolzanos_test(function, interval_inf, interval_cut):
            interval_sup = interval_cut
        elif bolzanos_test(function, interval_cut, interval_sup):
            interval_inf = interval_cut

    approx_root = half(interval_inf, interval_sup)

    if info:
        print(f"Iteration = {iteration}; x approx= {approx_root}; Interval inf = {interval_inf}; Interval sup = {interval_sup}; Interval size = {interval_sup - interval_inf}; f(x) approx = {function(approx_root):}\n")
        show_graph(function, initial_interval[0], initial_interval[1], approx_root)

    return approx_root


def iterations_bisection_method(interval_length, precision, info=False):
    if info:
        print(np.log(interval_length/precision)/np.log(2)+1)
    return int(np.ceil(np.log(interval_length/precision)/np.log(2)+1))


def false_position_method(function, interval_inf, interval_sup, epsilon_x=0, epsilon_y=0, iter_max=0, info=False):
    initial_interval = (interval_inf,interval_sup)
    iteration = 0
    axis_line_intersection = lambda a, b: (a * function(b) - b * function(a))/(function(b) - function(a))

    if info:
        print("False position method:")

    if (epsilon_x == epsilon_y == iter_max == 0) or interval_inf > interval_sup:
        if info:
            print("Wrong parameters")
        return

    if function(interval_inf) == 0:
        return interval_inf
    if function(interval_sup) == 0:
        return interval_sup

    if not bolzanos_test(function, interval_inf, interval_sup):
        if info:
            print("No roots in the interval according to bolzano's (cannot affirm anything)")
        return

    while abs(interval_sup - interval_inf) >= epsilon_x and abs(function(axis_line_intersection(interval_inf, interval_sup))) >= epsilon_y and (iter_max == 0 or iteration < iter_max):
        interval_cut = axis_line_intersection(interval_inf, interval_sup)

        if info:
            print(f"Iteration = {iteration}; x approx = {interval_cut}; Interval inf = {interval_inf}; Interval sup = {interval_sup}; Interval size = {interval_sup - interval_inf}; f(x) approx = {function(interval_cut)}")
        iteration += 1

        if function(interval_inf) * function(interval_cut) < 0:
            interval_sup = interval_cut
        elif function(interval_cut) * function(interval_sup) < 0:
            interval_inf = interval_cut

    approx_root = axis_line_intersection(interval_inf, interval_sup)

    if info:
        print(f"Iteration = {iteration}; x approx= {approx_root}; Interval inf = {interval_inf}; Interval sup = {interval_sup}; Interval size = {interval_sup - interval_inf}; f(x) approx = {function(approx_root):}\n")
        show_graph(function, initial_interval[0], initial_interval[1], approx_root)

    return approx_root


def fixed_point_method(function, phi_function, previous_x, epsilon_x=0, epsilon_y=0, iter_max=0, info=False):
    iteration = 0
    current_x = phi_function(previous_x)

    if info:
            print(f"Iteration = {iteration}; Current x = {previous_x}; f approximation = {function(previous_x)}")

    iteration +=1

    while abs(current_x - previous_x) >= epsilon_x and abs(function(current_x)) >= epsilon_y and (iter_max == 0 or iteration < iter_max):
        if info:
            print(f"Iteration = {iteration}; Current x = {current_x}; x difference = {abs(current_x - previous_x)}; f approximation = {function(current_x)}")

        previous_x = current_x
        current_x = phi_function(previous_x)
        iteration += 1

    approx_root = current_x
    if info:
        print(f"Iteration = {iteration}; Current x = {approx_root}; x difference = {abs(current_x - previous_x)}; f approximation = {function(current_x)}")
        show_graph(function, approx_root - 5, approx_root + 5, approx_root)

    return approx_root


def newton_method(function, function_diff, previous_x, epsilon_x=0, epsilon_y=0, iter_max=0, info=False):
    if info:
        print("Newton method:")
    phi = lambda x: x - function(x)/function_diff(x)
    return fixed_point_method(function, phi, previous_x, epsilon_x, epsilon_y, iter_max, info)


def secant_method(function, previous_x, previous_x2, epsilon_x=0, epsilon_y=0, iter_max=0, info=False):
    if info:
        print("Secant method:")
        print(f"x[0] = {previous_x}; x[1] = {previous_x2}")

    phi = lambda prev_x, x: (prev_x * function(x) - x * function(prev_x)) / (function(x) - function(prev_x))    
    current_x = phi(previous_x, previous_x2)
    iteration = 1

    while abs(current_x - previous_x) >= epsilon_x and abs(function(current_x)) >= epsilon_y and (iter_max == 0 or iteration < iter_max):
        if info:
            print(f"Iteration = {iteration}; Current x[{iteration+1}] = {current_x}; x difference = {abs(current_x - previous_x)}; f approximation = {function(current_x)}")

        previous_x2 = previous_x
        previous_x = current_x
        current_x = phi(previous_x, previous_x2)
        iteration += 1

    approx_root = current_x
    if info:
        print(f"Iteration = {iteration}; Current x[{iteration+1}] = {approx_root}; x difference = {abs(current_x - previous_x):e}; f approximation = {function(current_x):e}")
        show_graph(function, approx_root - 5, approx_root + 5, approx_root)

    return approx_root


def show_graph(function, interval_start, interval_finish, root='nope'):
    x = np.linspace(interval_start, interval_finish)
    plt.plot(x, function(x), color="purple")
    if root!='nope':
        plt.axvline(root, color="gray")
    plt.grid()
    plt.show()
