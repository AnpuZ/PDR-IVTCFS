# granulator.py
import numpy as np
from collections import defaultdict

def calculate_neighbourhood_radius_method_2(decision_table, decision_partition_set, num_object, num_attribute):
    
    current_neighbourhood_radius_result = np.zeros((num_object, num_attribute))
    attribute_type = [False] * num_attribute

    for i in range(num_attribute):
        column = decision_table[:, i]
        column_data = column.copy()
        unique_values = np.unique(column_data)

        if len(unique_values) <= 12:
            current_neighbourhood_radius_result[:, i] = 0
        else:
            attribute_type[i] = True
            for index, sublist in enumerate(decision_partition_set):
                if len(sublist) == 1:
                    value = sublist[0]
                    current_neighbourhood_radius_result[value, i] = 0
                else:
                    for subindex, value in enumerate(sublist):
                        homogeneous = sublist.copy()
                        homogeneous.remove(value)
                        new_array = column_data[homogeneous]
                        homogeneous_diffs = np.abs(new_array - column_data[value])
                        max_intra_class_distances = max(homogeneous_diffs) if len(homogeneous_diffs) > 0 else 0

                        heterogeneous = np.delete(column_data, sublist)
                        heterogeneous_diffs = np.abs(heterogeneous - column_data[value])
                        min_inter_class_distances = min(heterogeneous_diffs) if len(heterogeneous_diffs) > 0 else 0

                        if max_intra_class_distances <= min_inter_class_distances:
                            current_neighbourhood_radius_result[value, i] = max_intra_class_distances
                        else:
                            current_neighbourhood_radius_result[value, i] = max_intra_class_distances - min_inter_class_distances

    return current_neighbourhood_radius_result, attribute_type

def calculate_neighborhood_granule_method_4(attribute_matrix, attribute, neighbourhood_radius_result_2, attribute_type, num_object):
    
    t1_neighborhood_granule_set = []
    column_data = attribute_matrix[:, attribute]
    attribute_values = column_data.copy()
    
    if attribute_type[attribute]:
        for i in range(num_object):
            key_value = attribute_values[i]
            distance = np.abs(attribute_values - key_value)
            object_radius = neighbourhood_radius_result_2[i, attribute]
            neighborhood_granule_wrt_current_attribute = np.where(distance <= object_radius)[0]
            neighborhood_granule_wrt_attribute_i = neighborhood_granule_wrt_current_attribute.tolist()
            t1_neighborhood_granule_set.append(neighborhood_granule_wrt_attribute_i)
        
        for index, sublist in enumerate(t1_neighborhood_granule_set):
            for element in sublist.copy():
                if index in t1_neighborhood_granule_set[element]:
                    pass
                else:
                    t1_neighborhood_granule_set[index].remove(element)
    else:
        index_dict = defaultdict(list)
        for index, value in enumerate(attribute_values):
            index_dict[value].append(index)
        t1_neighborhood_granule_set = list(index_dict.values())
    
    return t1_neighborhood_granule_set

def calculate_neighborhood_about_attribute_subset_1(neighborhood_granule_set, current_neighborhood_granule, num_object):
    
    current_neighborhood_granule_set11 = []
    for index, sublist1 in enumerate(neighborhood_granule_set):
        sublist2 = current_neighborhood_granule[index]
        intersection = list(set(sublist1) & set(sublist2))
        current_neighborhood_granule_set11.append(intersection)
    return current_neighborhood_granule_set11

def calculate_neighborhood_about_attribute_subset_2(neighborhood_granule_set, current_neighborhood_granule, num_object):
    
    current_neighborhood_granule_set11 = []
    for sublist1 in neighborhood_granule_set:
        for sublist2 in current_neighborhood_granule:
            intersection = list(set(sublist1) & set(sublist2))
            if intersection:
                current_neighborhood_granule_set11.append(intersection)
    return current_neighborhood_granule_set11

def calculate_neighborhood_about_attribute_subset_3(neighborhood_granule_set, current_neighborhood_granule, num_object):
    
    current_neighborhood_granule_set11 = []
    for index, sublist1 in enumerate(neighborhood_granule_set):
        for sublist2 in current_neighborhood_granule:
            if index in sublist2:
                intersection = list(set(sublist1) & set(sublist2))
                current_neighborhood_granule_set11.append(intersection)
                break
    return current_neighborhood_granule_set11

def are_any_true(bool_array):
    return any(bool_array)

def neighborhood_about_conditional_attribute_set(matrix_A, neighbourhood_radius_result_2, current_attribute_type, num_object):
    
    type_of_selected_attribute = []
    neighborhood_granules_about_conditional_attribute_set = calculate_neighborhood_granule_method_4(
        matrix_A, 0, neighbourhood_radius_result_2, current_attribute_type, num_object)
    type_of_selected_attribute.append(current_attribute_type[0])
    
    num = matrix_A.shape[1]
    for i in range(1, num):
        current_neighborhood_granule = calculate_neighborhood_granule_method_4(
            matrix_A, i, neighbourhood_radius_result_2, current_attribute_type, num_object)
        
        if are_any_true(type_of_selected_attribute) and current_attribute_type[i]:
             neighborhood_granules_about_conditional_attribute_set = calculate_neighborhood_about_attribute_subset_1(neighborhood_granules_about_conditional_attribute_set, current_neighborhood_granule, num_object)
        elif are_any_true(type_of_selected_attribute) and not current_attribute_type[i]:
            neighborhood_granules_about_conditional_attribute_set = calculate_neighborhood_about_attribute_subset_3(
                neighborhood_granules_about_conditional_attribute_set, current_neighborhood_granule, num_object)
        elif current_attribute_type[i] and not are_any_true(type_of_selected_attribute):
            neighborhood_granules_about_conditional_attribute_set = calculate_neighborhood_about_attribute_subset_3(
                current_neighborhood_granule, neighborhood_granules_about_conditional_attribute_set, num_object)
        elif not current_attribute_type[i] and not are_any_true(type_of_selected_attribute):
            neighborhood_granules_about_conditional_attribute_set = calculate_neighborhood_about_attribute_subset_2(
                neighborhood_granules_about_conditional_attribute_set, current_neighborhood_granule, num_object)
        
        type_of_selected_attribute.append(current_attribute_type[i])
    return neighborhood_granules_about_conditional_attribute_set