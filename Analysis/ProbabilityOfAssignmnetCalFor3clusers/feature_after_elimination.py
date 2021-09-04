import numpy as np
import pandas as pd

def feature_after_elimination(data,X_names_train):
    '''
    input: data
    X_names_train: It contains features-names which are obtained after elimination of traning-set 
    how it works: 
    In data if columns are not in X_name then it will be removed 
    '''
    data_modified = data.copy()
    data_columns = data.columns
    for i,item in enumerate(data_columns):
        if (data_columns[i] not in X_names_train):
            data_modified = data_modified.drop(data_columns[i],axis=1)  #pandas array passed
    return data_modified
