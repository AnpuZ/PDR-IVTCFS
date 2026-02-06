# feature_selector.py
import numpy as np
from granulator import (
    calculate_neighborhood_granule_method_4,
    calculate_neighborhood_about_attribute_subset_1,
    calculate_neighborhood_about_attribute_subset_2,
    calculate_neighborhood_about_attribute_subset_3,
    are_any_true
)
from entropy_calculator import self_information_computation

def heuristic_algorithm_possibility_degree_2(decision_table, decision_matrix, attribute_matrix, attribute_cost, num_object, num_attribute, decision_classes,neighbourhood_radius_result_2, attribute_type, matrix_interval_comparison, cost_parameter, coverage_generated_by_neighborhood ):

    types = []
    selected_attributes = []
    self_information_of_selected_attributes = 0
    total_interval_value_test_cost = [0.0, 0.0]
    neighborhood_granule_set = []

    self_information_of_attribute_set = self_information_computation(decision_classes, coverage_generated_by_neighborhood)
    should_restart = True
    
    while should_restart:
        should_restart = False
        importance_function_increment = -np.inf
        optional_attribute = None
        current_optimal_neighborhood_granule_set = None 
        tem_optional_self_information = 0

        for i in range(num_attribute):
            if i not in selected_attributes:
                current_neighborhood_granule = calculate_neighborhood_granule_method_4(attribute_matrix, i,neighbourhood_radius_result_2, attribute_type, num_object)
                
                if len(selected_attributes) == 0:
                    current_neighborhood_granule_set = current_neighborhood_granule
                else:
                    if are_any_true(types) and attribute_type[i]:
                        current_neighborhood_granule_set = calculate_neighborhood_about_attribute_subset_1(
                            neighborhood_granule_set, current_neighborhood_granule, num_object)
                    elif are_any_true(types) and not attribute_type[i]:
                        current_neighborhood_granule_set = calculate_neighborhood_about_attribute_subset_3(
                            neighborhood_granule_set, current_neighborhood_granule, num_object)
                    elif attribute_type[i] and not are_any_true(types):
                        current_neighborhood_granule_set = calculate_neighborhood_about_attribute_subset_3(
                            current_neighborhood_granule, neighborhood_granule_set, num_object)
                    elif not attribute_type[i] and not are_any_true(types):
                        current_neighborhood_granule_set = calculate_neighborhood_about_attribute_subset_2(
                            neighborhood_granule_set, current_neighborhood_granule, num_object)
                
                self_information_about_current_attribute = self_information_computation(decision_classes, current_neighborhood_granule_set)
                
                # 计算Sig
                if len(selected_attributes) == 0:
                    sig_numerator = self_information_about_current_attribute
                else:
                    sig_numerator = self_information_of_selected_attributes - self_information_about_current_attribute
                
                if cost_parameter[i] == 0:
                    current_information_increment = sig_numerator / 0.001
                else:
                    current_information_increment = sig_numerator / cost_parameter[i]
                
                if current_information_increment > importance_function_increment:
                    importance_function_increment = current_information_increment
                    optional_attribute = i
                    current_optimal_neighborhood_granule_set = current_neighborhood_granule_set
                    tem_optional_self_information = self_information_about_current_attribute
        
        if optional_attribute is not None:
            selected_attributes.append(optional_attribute)
            total_interval_value_test_cost[0] += attribute_cost[optional_attribute, 0]
            total_interval_value_test_cost[1] += attribute_cost[optional_attribute, 1]
            self_information_of_selected_attributes = tem_optional_self_information
            neighborhood_granule_set = current_optimal_neighborhood_granule_set
            temp_type = attribute_type[optional_attribute]
            types.append(temp_type)
            
            if self_information_of_selected_attributes <= self_information_of_attribute_set + 1e-10:
                pass
            else:
                should_restart = True

    if len(selected_attributes) > 1:
        loop_completed = False 
        while not loop_completed:
            loop_completed = True
            current_selected_attributes = selected_attributes.copy()
            current_types = types.copy()
            
            for index, j in enumerate(current_selected_attributes):
                backup_1 = j
                
                temp_attr_list = current_selected_attributes[:index] + current_selected_attributes[index+1:]
                temp_type_list = current_types[:index] + current_types[index+1:]
                
                generated_neighborhood_granule_set = []
                types_2 = []

                for idx_m, m in enumerate(temp_attr_list):
                    new_neighborhood_granule = calculate_neighborhood_granule_method_4(attribute_matrix, m, neighbourhood_radius_result_2, attribute_type, num_object)
                    if not generated_neighborhood_granule_set:
                        generated_neighborhood_granule_set = new_neighborhood_granule
                    else:
                        if are_any_true(types_2) and attribute_type[m]:
                            generated_neighborhood_granule_set = calculate_neighborhood_about_attribute_subset_1(
                                generated_neighborhood_granule_set, new_neighborhood_granule, num_object)
                        elif are_any_true(types_2) and not attribute_type[m]:
                            generated_neighborhood_granule_set = calculate_neighborhood_about_attribute_subset_3(
                                generated_neighborhood_granule_set, new_neighborhood_granule, num_object)
                        elif attribute_type[m] and not are_any_true(types_2):
                            generated_neighborhood_granule_set = calculate_neighborhood_about_attribute_subset_3(
                                new_neighborhood_granule, generated_neighborhood_granule_set, num_object)
                        elif not attribute_type[m] and not are_any_true(types_2):
                            generated_neighborhood_granule_set = calculate_neighborhood_about_attribute_subset_2(
                                new_neighborhood_granule, generated_neighborhood_granule_set, num_object)
                    types_2.append(attribute_type[m])
                
                self_information_of_current_attribute_2 = self_information_computation(decision_classes,
                                                                                    generated_neighborhood_granule_set)

                if self_information_of_current_attribute_2 <= self_information_of_attribute_set + 1e-10:
                    selected_attributes.remove(j)

                    del types[index]
                    
                    total_interval_value_test_cost[0] -= attribute_cost[j, 0]
                    total_interval_value_test_cost[1] -= attribute_cost[j, 1]
                    loop_completed = False
                    break
                else:
                    pass

    total_interval_value_test_cost[0] = round(total_interval_value_test_cost[0], 2)
    total_interval_value_test_cost[1] = round(total_interval_value_test_cost[1], 2)

    return selected_attributes, total_interval_value_test_cost
