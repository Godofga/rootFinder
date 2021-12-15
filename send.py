def f(x,y):
    return -0.011*y - 0.02*x

def main_lop():
    max_temperature = 79
    orders = int(input())
    initial_temperatures = []
    cool_down_time = 0
    steps = 1

    for i in range(orders):
        temperature = float(input())
        initial_temperatures.append(temperature)

    for i in range(len(initial_temperatures)):
        while True:
            current_temperature = modified_euler_method_edo(f, cool_down_time, (0,initial_temperatures[i]), cool_down_time if cool_down_time!=0 else 1)
            if current_temperature < max_temperature:
                break
            cool_down_time += 1
    
    print(cool_down_time + 1 + initial_temperatures.index(max(initial_temperatures)))

    for i in range(len(initial_temperatures)):
        current_temperature = modified_euler_method_edo(f, cool_down_time, (0,initial_temperatures[i]), cool_down_time)
        print(round(current_temperature,4))

def f(x,y):
    return -0.011*y - 0.02*x

def main():
    max_temperature = 79
    orders = int(input())
    initial_temperatures = []
    cool_down_time = 0
    steps = 1
    greatest_time = 1

    for i in range(orders):
        temperature = float(input())
        initial_temperatures.append(temperature)

    for i in range(len(initial_temperatures)):
        cool_down_time = 0
        while True:
            current_temperature = modified_euler_method_edo(f, cool_down_time, (0,initial_temperatures[i]), max(cool_down_time,steps))
            if current_temperature < max_temperature:
                if cool_down_time + i + 1 > greatest_time:
                    greatest_time = cool_down_time + i + 1
                break
            cool_down_time += 1
        
    print(greatest_time)

    for i in range(len(initial_temperatures)):
        current_temperature = modified_euler_method_edo(f, greatest_time - i - 1, (0,initial_temperatures[i]), max(greatest_time - i - 1,steps))
        print(round(current_temperature,4))        


def modified_euler_method_edo(function, wanted_x, initial_conditions,  steps):
    x, y = initial_conditions
    H = (wanted_x - x)/steps

    for i in range(steps):
        y = y + H*(function(x, y) + function(x + H, y + H*function(x, y)))/2
        x += H
    
    return y
 
if __name__ == '__main__':
    main()

