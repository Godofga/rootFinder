import numpy as np
from scipy.linalg import lu, cholesky, norm


def input_matrix(dimension, b=False):
    dimension = int(dimension)
    
    matrix_a = np.zeros([dimension, dimension])
    matrix_b = np.zeros(dimension)
    for i in range(dimension):
        for j in range(dimension):
            matrix_a[i, j] = float(input())

    if b:
        for i in range(dimension):
            matrix_b[i] = float(input())
        return matrix_a, matrix_b
    return matrix_a


def check_line_dominance(matrix):
    matrix = np.copy(matrix)
    dom = 0
    for i in range(len(matrix)):
        summatory = 0
        line = list(matrix[i])
        line.sort()
        index = line.index(matrix[i, i])
        for j in range(len(matrix)):
            if j != index:
                summatory += abs(line[j])

        if abs(matrix[i, i]) <= summatory:
            dom = -1
            break

    operations = []
    if dom == -1:
        for i in range(len(matrix)):
            line = list(matrix[i])
            for j in range(len(line)):
                line[j] = abs(line[j])

            operations.append((i, line.index(max(line))))

        new_matrix = np.copy(matrix)
        for operation in operations:
            new_matrix[operation[1]] = matrix[operation[0]]

        matrix = np.copy(new_matrix)

        dom = 1
        for i in range(len(matrix)):
            summatory = 0
            line = list(matrix[i])
            line.sort()
            index = line.index(matrix[i, i])
            for j in range(len(matrix)):
                if j != index:
                    summatory += abs(line[j])

            if abs(matrix[i, i]) <= summatory:
                dom = 2
                break

    print("a matriz",
          "possui dominancia." if dom == 0 else "possui dominancia se trocar linhas:" if dom == 1 else "nao possui dominancia, mesmo se trocar linhas.")
    if dom == 1:
        for operation in operations:
            print(f"L{operation[0]} vai para L{operation[1]}")


def gaussian_elimination(matrix_a, matrix_b, pivot=False):
    matrix_a, matrix_b = np.copy(matrix_a), np.copy(matrix_b)
    operations = []
    for column in range(len(matrix_a)):
        if pivot and column < len(matrix_a)-1:
            matrix_a_list = list(matrix_a[column:, column])

            for i in range(len(matrix_a_list)):
                matrix_a_list[i] = abs(matrix_a_list[i])

            target = column + matrix_a_list.index(max(matrix_a_list))
            operations.append((column, target))
            switch_rows(matrix_a, column, target)
            switch_rows(matrix_b, column, target)

        for row in range(column + 1, len(matrix_a)):
            multiplier = -matrix_a[row, column] / matrix_a[column, column]
            matrix_a[row] += multiplier * matrix_a[column]
            matrix_b[row] += multiplier * matrix_b[column]
            matrix_a[row, column] = 0

    matrix_x = np.zeros(len(matrix_a))
    for i in range(len(matrix_x) - 1, -1, -1):
        matrix_x[i] = matrix_b[i]
        for j in range(i + 1, len(matrix_x)):
            matrix_x[i] -= matrix_a[i, j] * matrix_x[j]

        matrix_x[i] /= matrix_a[i, i]

    if pivot:
        for j in range(len(operations)-1, -1, -1):
            switch_rows(matrix_x, operations[j][0], operations[j][1])

    return matrix_a, matrix_b, matrix_x

def switch_rows(matrix, row_1, row_2):
    aux = np.copy(matrix[row_1])
    matrix[row_1] = np.copy(matrix[row_2])
    matrix[row_2] = np.copy(aux)


def gauss_jacobi(matrix_a, matrix_b,  error, err_min=0, err_max=0, iter_max=0, info=False):
    previous_x, current_x = np.zeros(len(matrix_a)), np.zeros(len(matrix_a))
    iteration = 0

    if info:
        print(f"Iteration = {iteration}, Error = {error(current_x)}, Current solution = {current_x}")

    if error(current_x) < err_min != 0 or iteration > iter_max != 0:
        return current_x, iteration

    while True:
        previous_x = np.copy(current_x)
        for i in range(len(current_x)):
            current_x[i] = matrix_b[i]
            for j in range(len(current_x)):
                if i != j:
                    current_x[i] -= matrix_a[i][j]*previous_x[j]

            current_x[i] /= matrix_a[i, i]

        iteration += 1
        if info:
            print(f"Iteration = {iteration}, Error = {error(current_x)}, Current solution = {current_x}")

        if error(current_x) < err_min != 0 or iteration > iter_max != 0:
            return current_x, iteration


def gauss_seidel(matrix_a, matrix_b,  error, err_min=0, err_max=0, iter_max=0, info=False):
    current_x = np.zeros(len(matrix_a))
    iteration = 0

    if info:
        print(f"Iteration = {iteration}, Error = {error(current_x)}, Current solution = {current_x}")

    if error(current_x) > err_max != 0 or error(current_x) < err_min != 0 or iteration > iter_max != 0:
        return current_x, iteration

    while True:
        previous_x = np.copy(current_x)
        for i in range(len(current_x)):
            current_x[i] = matrix_b[i]
            for j in range(len(current_x)):
                if i != j:
                    current_x[i] -= matrix_a[i][j]*current_x[j]

            current_x[i] /= matrix_a[i, i]

        iteration += 1
        if info:
            print(f"Iteration = {iteration}, Error = {error(current_x)}, Current solution = {current_x}")

        if error(current_x) > err_max != 0 or error(current_x) < err_min != 0 or iteration > iter_max != 0:
            return current_x, iteration

def lu_factorization(matrix_a, matrix_b): ## não fiz para resolver sistemas nem pivoteamento parcial
    matrix_b = np.copy(matrix_b)
    matrix_l = np.copy(matrix_a)
    matrix_u = np.copy(matrix_a)
    matrix_l = np.eye(len(matrix_a))

    for column in range(len(matrix_u)):
        target = 0;

        for row in range(column + 1, len(matrix_u)):
            multiplier = -matrix_u[row, column] / matrix_u[column, column]
            matrix_l[row,column] = -multiplier
            matrix_u[row] += multiplier * matrix_u[column]
            matrix_u[row, column] = 0

    return matrix_l, matrix_u

def pa_lu_factorization(matrix_a):
    return lu(matrix_a)

def cholesky_factorization(matrix_a):
    return cholesky(matrix_a)

def is_simetric(matrix):
    result = matrix == np.transpose(matrix)
    return result.all()

def inf_norm(matrix):
    return norm(matrix, ord='inf')

def first_norm(matrix):
    return norm(matrix,ord=1)




















