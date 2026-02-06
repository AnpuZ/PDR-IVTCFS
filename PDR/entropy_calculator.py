# entropy_calculator.py
import math

def self_information_computation_about_Ei(decision_class_of_Ei, set_of_neighborhood_granules):
    
    lower_approximation_set_about_Ei = set()
    upper_approximation_set_about_Ei = set()
    list1 = set(decision_class_of_Ei)
    
    for sublist2 in set_of_neighborhood_granules:
        sublist2_set = set(sublist2)
        if sublist2_set.issubset(list1):
            lower_approximation_set_about_Ei.update(sublist2_set)
        if sublist2_set.intersection(list1):
            upper_approximation_set_about_Ei.update(sublist2_set)
            
    count_of_upper_approximation_about_Ei = len(upper_approximation_set_about_Ei)
    count_of_lower_approximation_about_Ei = len(lower_approximation_set_about_Ei)
    
    if count_of_upper_approximation_about_Ei == 0:
        return 0 
        
    precision_about_Ei = count_of_lower_approximation_about_Ei / count_of_upper_approximation_about_Ei
    roughness_about_Ei = 1 - precision_about_Ei
    
    if precision_about_Ei == 0:
        self_information_about_Ei = 100
    else:
        self_information_about_Ei = - roughness_about_Ei * math.log(precision_about_Ei)
    return self_information_about_Ei

def self_information_computation(decision_class, set_of_neighborhood_granules):
    
    self_information_about_D = 0
    for sublist1 in decision_class:
        tem_value = self_information_computation_about_Ei(sublist1, set_of_neighborhood_granules)
        self_information_about_D += tem_value
    return self_information_about_D