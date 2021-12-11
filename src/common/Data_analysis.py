'''
To analyze the data manually and property boundwise
'''

##Directory to export the file of combination of different files
dir_path = './../'

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)


#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
        Main_folder_dir += dir_split[i] + str('/')
    


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from search_fileNcreate import search_fileNcreate as SF

class Data_analysis():
    '''
    This class is useful for data analysis for available data.
    '''
    def __init__(self):
        pass

    def View_n_Analyze(file_location,current_directory):
        '''
        View and Analyze the data 
        '''
        df = pd.read_csv(str(file_location))
        print('df: ', df)
        # print('\n\nParameter associated with the dataset are : \n')
        # print(list(df.columns))

        print('\n\n\nFuels in the dataset:')
        Unique_Fuels = df['Fuel'].unique()
        for i,item in enumerate(Unique_Fuels):
            print(i ,':',str(Unique_Fuels[i]),'\n')
        
        while(1):
            Fuel_selection = input('Which fuel you want to analyse ? ')
            if(int(Fuel_selection) in range(len(Unique_Fuels))):
                break
            else:
                print('Please select appropriate number. \n')
                
        #Comparing the given choice with dataset and filtering out the value 
        df_fuel = df[df['Fuel'] == str(Unique_Fuels[int(Fuel_selection)])]
        # df_fuel.to_csv('df_fuel.csv')
        # print('df_fuel: ', df_fuel)

        ##Note : Fixing pressure intially generates problem as it continuos number.  
        ##        To find out specific range of pressure and related unique equivalence ratio 
        ##        of the experimental data is quite intricate. 
        ##        Better approach is to find out unique equivalence ratio and then find out pressure range

        equivalence_ratio_available = df_fuel['Equv(phi)'].unique()
        

        ###Fixing Up Equivalence ratio 
        while(1):
            print('\n \n Available equivalence ratio for selected fuel are given as below : \n')
            for i,item in enumerate(equivalence_ratio_available):
                print(equivalence_ratio_available[i])
            equivalence_ratio_choice = input('\n \n Fixing the Equivalence ratio value to analyse the data. \n What is your value for Equivalence ratio ? \n')

            if(float(equivalence_ratio_choice) in equivalence_ratio_available):
                break
            else:
                print('Please select pressure from the available data only. \n')

        ##Filtering out the data by given choice of equivalence ratio 
        df_equivalence_ratio = df_fuel[df_fuel['Equv(phi)'] == float(equivalence_ratio_choice)]
        # print('df_equivalence_ratio: ', df_equivalence_ratio)

        ####To fixing up the parameters
        print('\n \n Associated pressure with selected Fuel: \n')
        All_Pressure = df_equivalence_ratio['P(atm)']
        All_Pressure = All_Pressure.sort_values() # Sorted the pressure value 
        # print('All_Pressure: ', All_Pressure)


        ## trying to find out minimum choices of available by rounding off the numbers and 
        ## trying to find out the unique numbers out of that 
        Available_pressure_choices = []
        for i in range(len(All_Pressure)):
            Available_pressure_choices.append(round(list(All_Pressure)[i]))  #Converted into the list to avoid index problem
        unique_pressure_choices = list(set((Available_pressure_choices))) #Available pressure choices 
        # print('unique pressure_choices: ', unique_pressure_choices)

        # print('All_Pressures: \n', All_Pressure)
        # print('length of All_Pressure: ', len(All_Pressure))
        print('Avilable Pressure choices : ', unique_pressure_choices)

        ###Fixing Up pressure 
        while(1):
            print('Maximum Pressure: ', df_equivalence_ratio['P(atm)'].max())
            print('Minimum Pressure: ', df_equivalence_ratio['P(atm)'].min())
            print('Available Pressure Choices are as below : \n')
            for i,item in enumerate(unique_pressure_choices):
                print(str(unique_pressure_choices[i]))
            pressure_choice = input('\n \n Fixing the pressure value to analyse the data. \n What is your value for pressure ? \n')

            if(float(pressure_choice) in unique_pressure_choices):
                break
            else:
                print('Please select pressure from the avilable data only. \n')

        

        #Uncertainty considered as slight deviation in pressure occurs in shock tube 
        Uncertainty_in_Pressure = .5
        uncertainty_choice = input ('\n Uncertainty in pressure is considered as + or - 0.5 atm. \n Do you want to change it ? y or n : ')
        if(uncertainty_choice == 'y'):
            Uncertainty_in_Pressure = input('Enter your uncertainty value ?')
        
        Upper_limit_of_pressure = float(max(pressure_choice)) + float(Uncertainty_in_Pressure) #Pressure upper limit for consideraion in dataset 
        print('Upper_limit_of_pressure: ', Upper_limit_of_pressure)
        Lower_limit_of_pressure = float(min(pressure_choice)) - float(Uncertainty_in_Pressure)  #Pressure lower limit to consideraion in dataset
        print('Lower_limit_of_pressure: ', Lower_limit_of_pressure)

        ##Filtering out the data by upper and lower limit 
        df_pressure = df_equivalence_ratio[df_equivalence_ratio['P(atm)'] > Lower_limit_of_pressure]
        df_pressure = df_equivalence_ratio[df_equivalence_ratio['P(atm)']  < Upper_limit_of_pressure]
        # df_pressure.to_csv('df_presure.csv')
        # print('df_pressure: ', df_pressure)

        #Plotting
        SF.check_directory(str(current_directory)+'/result/AnalysisResult/')
        plt.scatter(1000/df_pressure['T(K)'],np.log(df_pressure['Time(μs)']),label='Fuel:'+str(Unique_Fuels[int(Fuel_selection)])+'  $\phi :$'+str(equivalence_ratio_choice)+'  Pressure Range :{'+str(Lower_limit_of_pressure)+','+str(Upper_limit_of_pressure)+'}')
        plt.xlabel('1000/Temperature')
        plt.ylabel('Time(μs)')
        # plt.rc('text', usetex=True)        
        plt.legend()
        plt.savefig(str(current_directory)+'/result/AnalysisResult/fuel_analysis.png')
        plt.show()





        


        

        
