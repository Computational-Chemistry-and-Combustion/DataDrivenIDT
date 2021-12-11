############################ FInd_Fuel_Name Module ##################################

dir_path = './../'

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)


#Obtaining Path of directory 
dir_split = dir_path.split('/')
Main_folder_dir = ''
for i in range(len(dir_split)-1):
        Main_folder_dir += dir_split[i] + str('/')
    


# Importing the libraries
import numpy as np
import pandas as pd

def Fuels_name_generation(list_fuel):
        '''
        Fuel name genearation takes an array of fuels(SMILES) and based on array it finds of fuel name. 
        Essentially useful for printing  fuel_names. 
        It uses FUEL_NAME.csv as reference which is in data folder.
        - For ease of identification of fuels when combination is made
        Arguments: (list_fuel)

        Pass the list of fuels(SMILES) as array if those SMILES are avialble in the database(FUEL_NAME.csv)
        it will return array of actual combination of fuel_name array.
        '''
        ###Dictionary of SMILE name 
        Fuel_Name = pd.read_csv(str(Main_folder_dir)+'/data/Fuel_Name.csv',header = None)
        keys = Fuel_Name[0]
        values = Fuel_Name[1]
        Smile_name_directory  = dict(zip(keys, values))
        # print('Smile_name_directory: ', Smile_name_directory)
        Fuel_Name = Fuel_Name.iloc[0,0]         #Emptying the dataset

        Fuels_combo_name =''

        for i,item in enumerate(list_fuel):
                Fuels_combo_name = Fuels_combo_name +str('_N_') + str(Smile_name_directory.get(list_fuel[i]))
        return Fuels_combo_name
