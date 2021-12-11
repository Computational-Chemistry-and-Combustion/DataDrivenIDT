import pandas as pd
import numpy as np
import regression
from search_fileNcreate import search_fileNcreate as SF
##Directory to export the file of combination of different files
dir_path = './../'
import time
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


def writing_coefficient(coefficient_dict,training_adj_r2,testing_Adj_r2,cluster_label,curr_directory,child_type='root'):
        dummy_data = pd.DataFrame()


        headers = list(coefficient_dict.keys())
        headers.extend(['Training_R2','Testing_R2'])

        #checking and creating directory 
        SF.check_directory(str(curr_directory)+'/result/Tree_coefficient_result/'+str(child_type)+'/')
        SF.check_file_existence(str(curr_directory)+'/result/Tree_coefficient_result/'+str(child_type)+'/cluster_'+str(cluster_label)+'.csv')
        #file handling  and saving result 
        try:
            #if file exist it will read and append the output 
            df =  pd.read_csv(str(curr_directory)+'/result/Tree_coefficient_result/'+str(child_type)+'/cluster_'+str(cluster_label)+'.csv')    #reading dataset 
        except pd.errors.EmptyDataError:
            #if file doesn't exist it will create empty dataframe and append the output only with headers
            df = pd.DataFrame([],columns=headers)   #making dataframe with headers
            df = df[0:0]        #cleaning dataset 
            df.to_csv(str(curr_directory)+'/result/Tree_coefficient_result/'+str(child_type)+'/cluster_'+str(cluster_label)+'.csv',index = False)    #saving dataframe
            df =  pd.read_csv(str(curr_directory)+'/result/Tree_coefficient_result/'+str(child_type)+'/cluster_'+str(cluster_label)+'.csv')    #again reding csv to store result 

        #coefficient processing
        data_coefficient_keys = coefficient_dict.keys()
        data_coefficient_values = coefficient_dict.values()
        coefficient_series = []        #to appended in dataset
        for i in range(len(headers)-2): #-2 as last entry later added ###traing and testing result
            if(headers[i] in data_coefficient_keys):
                     coefficient_series.append(coefficient_dict.get(headers[i]))   #Appending corresponding value

        coefficient_series.append(training_adj_r2)
        coefficient_series.append(testing_Adj_r2)
        # print('coefficient_series: ', coefficient_series)
        df1 = pd.DataFrame([coefficient_series],columns=headers)

        # coefficient_series = pd.Series(coefficient_series)
        df = df.append(df1)
        df.to_csv(str(curr_directory)+'/result/Tree_coefficient_result/'+str(child_type)+'/cluster_'+str(cluster_label)+'.csv',index = False)    #saving dataframe