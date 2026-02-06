# cost_generator.py
import numpy as np
import random

def generate_random_number():
    return round(random.uniform(0, 1), 2)

def generate_interval_cost_list(number_of_conditional_attributes):
    
    interval_cost_list = []
    
    for _ in range(number_of_conditional_attributes):
        r1 = generate_random_number()
        r2 = generate_random_number()
        while r1 == r2:
            r2 = generate_random_number()
        interval = [min(r1, r2), max(r1, r2)]
        interval_cost_list.append(interval)
    
    return np.array(interval_cost_list)

def generate_cost_set_for_experiment(num_experiments, num_attributes):

    cost_set = []
    
    for i in range(num_experiments):
        cost_set.append(generate_interval_cost_list(num_attributes))
    
    return np.array(cost_set)