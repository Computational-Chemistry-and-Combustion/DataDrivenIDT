#########################  Regression  File  ##############################################
# This  file to grab the required columns for the dataset then by processing 
# the columns convert into required format, training - testing split, model learniong 
# and backward elimination
#############################################################################################

import numpy as np
import pandas as pd 
import time 
import random
import copy
##Directory to export the file of combination of different files
dir_path = './../'

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
# print('dir_path: ', dir_path)
sys.path.append(dir_path)


#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
        Main_folder_dir += dir_split[i] + str('/')

def equalize_dataset(dataset,choice_value):
        '''
        This method will find out the least number of data available for specific fuel out of all fuel 
        and based on that least number of it will generate the dataset, in which number of entries(data)
        of different of all fuels is same and equal to that least number. so dataset is equivalized with 
        equal fuel data in dataframe.
        Along with that, it also generates the dataset which contains entries which are in dataset but not 
        in Equivalized Dataset.
        Arguments: (dataset)

        dataset : pass the dataset. Note: datset should be in specified format as in data folder 
                  it will return the equivalized dataset.
        
        Return : 
        equal_entries_data , Diff_dataset  
        '''
        #Finding out the unique values of fuel 
        unique_fuel = list(dataset.Fuel.unique())

        from collections import Counter
        repetition_of_fuel = Counter(dataset['Fuel']) #Dictionary 
        min_fuel_data = min(list(repetition_of_fuel.values()))         #number of least data

        equal_entries_data = pd.DataFrame()
        #Generating dataset of equal number of fuel entries
        # print('min_fuel_data: ', min_fuel_data)
        for i,item in enumerate(unique_fuel):
                sub_dataset = dataset.loc[dataset['Fuel'] == unique_fuel[i]]
                sub_dataset = sub_dataset.sample(n=min_fuel_data)#, random_state=41)
                equal_entries_data = equal_entries_data.append(sub_dataset)

        #Difference between two dataset 
        Diff_dataset = pd.DataFrame()

        dataset_index_list = list(dataset.index)    #index list of main dataset 
        equal_entries_data_index_list = list(equal_entries_data.index)  #index list of equalised dataset 
        #Diff  between two list obtained by converting into set 
        diff_list = list(set(dataset_index_list) - set(equal_entries_data_index_list))  
        # Getting the differences of two Dataframe by index_diff
        for i,item in enumerate(diff_list):
                Diff_dataset = Diff_dataset.append(dataset.loc[diff_list[i],:])
        
        # equal_entries_data.to_csv('equal.csv')
        # Diff_dataset.to_csv('dff.csv')
        return equal_entries_data , Diff_dataset    
