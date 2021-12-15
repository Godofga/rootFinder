##not updated
import numpy as np
import string


def decimal_to_base_n(user_input, base):
    base = int(base)
    decimal = int(user_input)
    base_n = ""
    if decimal == 0:
        return '0'
    while decimal >0:
        rest = decimal % base
        decimal -= rest
        decimal = int(decimal / base)

        if(rest >=10):
            rest = string.ascii_uppercase[rest-10]

        base_n = str(rest) + base_n

    return base_n


def decimal_to_binary(user_input):
    return decimal_to_base_n(user_input, 2)


def decimal_to_hexadecimal(user_input):
    return decimal_to_base_n(user_input, 16)    


def base_n_to_decimal(user_input, base):
    base_n = user_input
    decimal = 0
    index = len(base_n)-1
    for character in base_n:
        if character in string.ascii_uppercase:
            decimal += (string.ascii_uppercase.find(character)+10) * base ** index
        else:
            decimal += int(character) * 16 ** index
        index -= 1
    return decimal


def hexadecimal_to_decimal(user_input):
    return base_n_to_decimal(user_input, 16)


def hexadecimal_to_binary(user_input):
    decimal = hexadecimal_to_decimal(user_input)
    return decimal_to_binary(decimal)


def binary_to_decimal(user_input):
    return base_n_to_decimal(user_input, 2)


def binary_to_base_n(user_input, base):
    decimal = binary_to_decimal(user_input)
    return decimal_to_base_n(decimal, base)


def binary_to_float(user_input):
    index = user_input.find('.')
    result = binary_to_decimal(user_input[:index])
    i=-1
    for digit in user_input[index+1:]:
        result += int(digit) * 2 ** i
        i-=1

    return result


def float_to_fbinary(user_input, digits):
    index = user_input.find('.')
    f_binary = decimal_to_binary(user_input[:index])
    f_binary += '.'
    floating_part = float("0" + user_input[index:])
    for i in range(digits):
        floating_part *= 2
        f_binary += str(int(np.floor(floating_part)))
        floating_part -= np.floor(floating_part)
    
    return f_binary

def float_to_fbase_n(user_input, base, digits):
    index = user_input.find('.')
    fbase_n = decimal_to_base_n(user_input[:index], base)
    fbase_n += '.'
    floating_part = float("0" + user_input[index:])
    for i in range(digits):
        floating_part *= base
        fbase_n += str(int(np.floor(floating_part)))
        floating_part -= np.floor(floating_part)
    
    return fbase_n


class machine:
   
    def __init__(self, base, n_mantissa, lower, upper):
        self.base = base
        self.n_mantissa = n_mantissa
        self.upper = upper
        self.lower = lower
   
    def highest_number(self, info=False):
        mantissa = str(int(self.base)-1)+ '.'
        for i in range(self.n_mantissa-1):
            mantissa +=str(int(self.base)-1)

        mantissa = float(mantissa)
        if info:
            print(f"Highest: {mantissa} * {self.base}^{self.upper} = {mantissa*self.base ** self.upper}")
        return mantissa*self.base ** self.upper

    def lowest_number(self, info=False):
        mantissa = 1.
        if info:
            print(f"Lowest: {mantissa} * {self.base}^{self.lower} = {mantissa*self.base ** self.lower}")
        return mantissa*self.base ** self.lower

    def underflow(self, number):
        return abs(number)<abs(self.lowest_number()) and number !=0
    
    def overflow(self, number):
        return abs(number)>abs(self.highest_number()) and number !=0

    def number_of_representations(self):
        return 2 * (self.base-1) * self.base **(self.n_mantissa -1) * (self.upper - self.lower +1) +1
 

def absolute_error(exact_value, aproximated_value):
    return abs(exact_value-aproximated_value)

def relative_error(exact_value, aproximated_value):
    return abs(exact_value-aproximated_value)/abs(exact_value)
    
        

#1239ABC92138
#20038904521016
#20038904521016