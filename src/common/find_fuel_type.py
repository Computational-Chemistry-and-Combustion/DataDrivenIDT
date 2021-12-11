import sys
import os 
import pandas as pd
dir_path = os.path.dirname(os.path.realpath(__file__))
# print('dir_path: ', dir_path)
sys.path.append(dir_path)


#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
    Main_folder_dir += dir_split[i] + str('/')


class find_fuel_type():
    

    def find_branched_alkanes(Fuel_Name_data):
        '''
        This module will find only stright chain alkanes from the dataset
        and it 
        '''
        unique_fuels = list(Fuel_Name_data['Fuel'].unique())   #findind out all unique fuels               
        list_fuel = []  #List of fuels to store

        # print(unique_fuels)
        for i,item in enumerate(unique_fuels):
            fuel_selected = unique_fuels[i]
            if('('  in fuel_selected):
                flag = 0
                for j,item in enumerate(fuel_selected):
                    if(fuel_selected[j] == 'C' or fuel_selected[j] == '(' or fuel_selected[j] == ')'):
                        flag = 0
                        continue
                    else:
                        flag = 1
                        break
                if(flag == 0):
                    list_fuel.append(fuel_selected)
        return list_fuel

    def find_strightchain_alkanes(Fuel_Name_data):
        '''
        This module will find only stright chain alkanes from the dataset
        and it 
        '''
        unique_fuels = list(Fuel_Name_data['Fuel'].unique())   #findind out all unique fuels               
        list_fuel = []  #List of fuels to store

        # print(unique_fuels)
        for i,item in enumerate(unique_fuels):
            fuel_selected = unique_fuels[i]
            flag = 0
            for j,item in enumerate(fuel_selected):
                if(fuel_selected[j] == 'C'):
                    flag = 0
                else:
                    flag = 1
                    break
            if(flag == 0):
                list_fuel.append(fuel_selected)
        
        return list_fuel
    
    def all_alkanes(Fuel_Name_data):
        '''
        This module will find only stright chain alkanes from the dataset
        and it 
        '''
        unique_fuels = list(Fuel_Name_data['Fuel'].unique())   #findind out all unique fuels               
        return unique_fuels

    def manual_selection(Fuel_Name_data):
        print("::::::::::::::::: Enter Your Choice of fuels ::::::::::::::::: \n")
        
        list_fuel = []  #List of fuels
        list_choice_number = [] #list of choice number
        
        avialable_fuel = Fuel_Name_data['Fuel'].unique()
        print("Avilable Choice: \n",)
        for i,item in enumerate(avialable_fuel):
            print(i,':',avialable_fuel[i],'\n')

        print("Enter x(SMALL) to end the list \n")

        
        #Input and print of the choice 
        input_given = 0
        while True:
            input_given = input('Enter Fuel choices :: \t ')
            if(input_given == 'x'):
                break
            if(int(input_given) > len(avialable_fuel)-1):
                print('Choice exceeds. Please check avilability')
                continue
            if(input_given.isnumeric() is False ):
                print('Please Enter Valid Input')
                continue
            if(input_given in list_choice_number):
                print('You have already selected the Fuel, Try different Fuel')
                continue
            list_fuel.append(avialable_fuel[int(input_given)])
            list_choice_number.append(input_given)
            print("Choices selected:")
            for i,item in enumerate(list_fuel):
                print(i, " :" ,list_fuel[i])#,int(list_fuel[i])])
        
        return list_fuel
    
    def find_strightchain_alkanes_dataset(Fuel_Name_data):
        '''
        This module will find only stright chain alkanes from the dataset
        and it 
        '''
        unique_fuels = list(Fuel_Name_data['Fuel'].unique())   #findind out all unique fuels               
        list_fuel = []  #List of fuels to store

        # print(unique_fuels)
        for i,item in enumerate(unique_fuels):
            fuel_selected = unique_fuels[i]
            flag = 0
            for j,item in enumerate(fuel_selected):
                if(fuel_selected[j] == 'C'):
                    flag = 0
                else:
                    flag = 1
                    break
            if(flag == 0):
                list_fuel.append(fuel_selected)
        another_data = pd.DataFrame()
        data = copy.deepcopy(Fuel_Name_data)
        for i,item in enumerate(list_fuel):
            data_to_append = data[data['Fuel'] == list_fuel[i]]
            another_data = another_data.append(data_to_append)
        
        return another_data