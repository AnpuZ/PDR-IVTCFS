# main.py
import warnings
from data_loader import load_and_preprocess_data, partition_decision_classes
from cost_generator import generate_cost_set_for_experiment
from ranking_strategy import ranking_method_of_probability_degree
from granulator import calculate_neighbourhood_radius_method_2, neighborhood_about_conditional_attribute_set
from feature_selector import heuristic_algorithm_possibility_degree_2

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    
    FILE_PATH = 'zoo_processed.csv'
    EXPERIMENT_TIMES = 10
    
    try:
        print(f"Loading data...: {FILE_PATH}")
        decision_table, decision_matrix, attribute_matrix, num_object, num_attribute = load_and_preprocess_data(FILE_PATH)
        print(f"|U|: {num_object}, |C|: {num_attribute}")
        
        print("Initialization calculation is in progress...")
        cost_set = generate_cost_set_for_experiment(EXPERIMENT_TIMES, num_attribute)    
        decision_classes = partition_decision_classes(decision_matrix)
        
        neighbourhood_radius_result_2, attribute_type = calculate_neighbourhood_radius_method_2(
            decision_table, decision_classes, num_object, num_attribute)
            
        coverage_generated_by_neighborhood = neighborhood_about_conditional_attribute_set(
            attribute_matrix, neighbourhood_radius_result_2, attribute_type, num_object)
        
        REDCT = []
        TOTAL_COST = []
        
        print("-" * 60)
        for i in range(EXPERIMENT_TIMES):
            print(f"The {i+1}/{EXPERIMENT_TIMES} experiment is currently underway...")
            attribute_cost = cost_set[i]
            
            matrix_interval_comparison, cost_parameter = ranking_method_of_probability_degree(attribute_cost, num_attribute)
            
            reduct, total_cost = heuristic_algorithm_possibility_degree_2(
                decision_table, decision_matrix, attribute_matrix, attribute_cost, 
                num_object, num_attribute, decision_classes,
                neighbourhood_radius_result_2, attribute_type, 
                matrix_interval_comparison, cost_parameter, coverage_generated_by_neighborhood
            )
            
            REDCT.append(reduct)
            TOTAL_COST.append(total_cost)
            print(f" The end of: reduct_len={len(reduct)}, total test cost={total_cost}")

        # 4. 结果汇总
        print("-" * 60)
        print("REDCT:", REDCT)
        print("TOTAL_COST:", TOTAL_COST)

    except FileNotFoundError:
        print(f"Error: Unable to find the file {FILE_PATH}.")
    except Exception as e:
        print(f"An unknown error has occurred:{e}")
        import traceback
        traceback.print_exc()