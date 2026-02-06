# data_loader.py
import pandas as pd
import numpy as np
from collections import defaultdict

def load_and_preprocess_data(file_path):

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format.")

    decision_table = np.array(df)
    
    attribute_matrix = decision_table[:, :-1].astype(float)

    decision_matrix = decision_table[:, -1]
    
    num_object, num_attribute = attribute_matrix.shape
    
    return decision_table, decision_matrix, attribute_matrix, num_object, num_attribute

def partition_decision_classes(decision_attribute):
    
    classification_results = defaultdict(list)
    
    for idx, decision in enumerate(decision_attribute):
        classification_results[decision].append(idx)
    
    return list(classification_results.values())