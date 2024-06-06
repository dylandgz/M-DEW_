from itertools import product
import os

import numpy as np
import pandas as pd



def get_dataset_characteristics(dataset_path):
    df = pd.read_csv(dataset_path)
    n_samples, n_cols = df.shape

    # ...

    return n_cols, n_samples



def get_correlation_distribution(df: pd.DataFrame):
    relevant_entries = [
        (x, y) 
        for x, y in product(range(df.shape[0]), range(df.shape[1]))
        if x < y
    ]
    corr_df = df.corr()
    correlations = [corr_df.iloc[relevant_entries[i]] for i in range(len(relevant_entries))]
    
