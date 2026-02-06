# ranking_strategy.py
import numpy as np

def compute_possibility_degree(list_1, list_2):

    len_a = list_1[1] - list_1[0]
    len_b = list_2[1] - list_2[0]

    temp_value = len_a + len_b - max(list_2[1] - list_1[0], 0)
    
    if len_a + len_b == 0:
        return 0
    
    possibility_degree = max(0, temp_value / (len_a + len_b))

    return possibility_degree

def ranking_method_of_probability_degree(two_dimensional_list, num_attribute):

    matrix_of_possibility_degree = np.zeros((num_attribute, num_attribute))
    sorting_vector = np.zeros(num_attribute)
    
    for i, sublist_1 in enumerate(two_dimensional_list):
        for j, sublist_2 in enumerate(two_dimensional_list):
            if i < j:
                matrix_of_possibility_degree[i, j] = compute_possibility_degree(sublist_1, sublist_2)
                matrix_of_possibility_degree[j, i] = 1 - matrix_of_possibility_degree[i, j]
            elif i == j:
                matrix_of_possibility_degree[i, j] = 0.5
                
    matrix_of_possibility_degree = np.round(matrix_of_possibility_degree, decimals=3)
    
    for j in range(num_attribute):

        sum_row = np.sum(matrix_of_possibility_degree[j, :])
        
        sorting_vector[j] = (sum_row + num_attribute/2 - 1) / (num_attribute * (num_attribute - 1))
        
    sorting_vector = np.round(sorting_vector, decimals=3)

    return matrix_of_possibility_degree, sorting_vector