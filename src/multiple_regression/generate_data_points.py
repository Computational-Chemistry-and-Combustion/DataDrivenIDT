##Directory to export the file of combination of different files
dir_path = './../'

import sys
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 

dir_path = os.path.dirname(os.path.realpath(__file__))
# print('dir_path: ', dir_path)
sys.path.append(dir_path)


#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
        Main_folder_dir += dir_split[i] + str('/')
    

    

class generate_data_points():
    '''
    transferred (passed dataset as argument) dataset it will generate more data points using uncertainity and appened 
    those data points to the transferred data set.  Data points are generated from each sample from transferred dataset. 
    it genrate 3 dimentional data
    '''
   
    def generate_dataset(data,choice_value):

        #unique_fuels list to process
        unique_fuels = list(data['Fuel'].unique())
        unique_fuels_data_count = []     #count of data points for diff unique fuel
        unique_fuel_count = len(unique_fuels)

        #fuel_list and number of data points 
        for i,item in enumerate(unique_fuels):
            unique_fuels_data_count.append(list(data['Fuel']).count(unique_fuels[i]))



        min_count_in_unique_fuel = min(unique_fuels_data_count)
        max_count_in_unique_fuel = max(unique_fuels_data_count)


        unique_fuel_zip = zip(unique_fuels,unique_fuels_data_count)
        unique_fuels_dict = dict(unique_fuel_zip)
        for i,item in enumerate(unique_fuels):
            print(i,':  ',unique_fuels[i], ':' , unique_fuels_data_count[i])

        # '''
        # algorithm:
        # 1. find out available data points  e.g::  fuel1:200, fuel2:500, fuel3:20, fuel4:70
        # 2. Find out max data points in the array e.f max_data_points = 500
        # 3. dataset size  =  max_data_points  * no_of_fuels e.g 500 * 4 = 2000
        # or ask them to change  but greater than this

        # 4. data to be generated from single data points wil be,
        #     datacount_from_single_data  = dataset_size / avialble_data_for_that_fuel 
        #     eg . for fuel2 ::: (2000/500) = 4 data points from single point
        #     so, number data points generated will be atleast no_of fuel_times from single point

        # 5. generate 2000 data points using uncertanity .....by mean and covariance matrix(diagonal as assumed that pressure and variacne are independent variable)

        # 6. if uncertainity is Nan then take least value from the that SPECIFIC fuel dataset

        # 7. pick datacount_from_single_data number of data points from the 

        # '''

        #number of data points for eahc fuel 
        no_data_to_generate = max_count_in_unique_fuel * unique_fuel_count * 1
        print('\nProcess will generate dataset of size  ',no_data_to_generate , ' data points ')

        y_or_n = 'y'
        if(choice_value == '16' or choice_value == '14' or choice_value == '15' ):
            y_or_n = 'n'
           
        if(y_or_n == 'y'):
            no_data_to_generate  =  int(input('What SIZE OF DATASET you want to generate for EACH FUEL?'))
            
        ###########
        # Pressure#
        ###########
        #least value in uncertainity of pressure 
        data.loc[:,'P_Error(%)'] = data.loc[:,'P_Error(%)'].replace(np.nan, '', regex=True)
        data.loc[:,'P_Error(%)'] = data.loc[:,'P_Error(%)'].replace('±',"",regex=True)
        data.loc[:,'P_Error(%)'] = data.loc[:,'P_Error(%)'].str.rstrip()
        #converting into numeric 
        data.loc[:,'P_Error(%)'] = pd.to_numeric(data['P_Error(%)'], errors='coerce')
        # print(data['P_Error(%)'].dtype)
        least_P_uncer = data.loc[:,'P_Error(%)'].min()
        #REPLACING NAN VALUE WITH LEAST VALUE
        data.loc[:,'P_Error(%)'] = data.loc[:,'P_Error(%)'].fillna(least_P_uncer)
        # print(data['P_Error(%)'])
        # print('least_P_uncer: ', least_P_uncer)

        ##############
        # temperature#
        ##############
        #least value in uncertainity of temp 
        data.loc[:,'T_Error(%)'] = data.loc[:,'T_Error(%)'].replace(np.nan, '', regex=True)
        data.loc[:,'T_Error(%)'] = data.loc[:,'T_Error(%)'].replace('±',"",regex=True)
        data.loc[:,'T_Error(%)'] = data.loc[:,'T_Error(%)'].str.rstrip()
        maxim = (data['T_Error(%)'].values)
        #CONVERTING DATA INTO NUMERIC IF BLANK IGNORE
        data.loc[:,'T_Error(%)'] = pd.to_numeric(data['T_Error(%)'], errors='coerce')
        least_T_uncer = data.loc[:,'T_Error(%)'].min()
        #REPLACING NAN VALUE WITH LEAST VALUE
        data.loc[:,'T_Error(%)'] = data.loc[:,'T_Error(%)'].fillna(least_P_uncer)

        # data.to_csv('data.csv')
        #final Extended dataframe
        extended_dataframe = data[0:0]


        for i,item in enumerate(unique_fuels):
            selected_fuel = unique_fuels[i]
            # print('selected_fuel: ', selected_fuel)

            #filter data using selected fuel data 
            specific_fuel_dataset  =  data[data.loc[:,'Fuel'] == selected_fuel]

            #resetting index 
            specific_fuel_dataset = specific_fuel_dataset.reset_index()

            # print('specific_fuel_dataset: ', specific_fuel_dataset)

            #specifc_fuel_count
            selected_fuel_data_count  = len(specific_fuel_dataset)
            # print('selected_fuel_data_count: ', selected_fuel_data_count)
            
        
            #number data points from each data point 
            data_generation_count = int(no_data_to_generate / selected_fuel_data_count)
            
            #if data point is less then 

            while(data_generation_count < 1):
                print('\n Please eneter number bigger than given as dataset has already more points than gien ')
                print('\n It should be at least two times of :',least_count_in_unique_fuel * unique_fuel_count)
                print('\n You have to enter size of FINAL Dataset')
                no_data_to_generate  =  int(input('What SIZE OF DATASET you want to generate for EACH FUEL?'))
                data_generation_count = int(no_data_to_generate  / selected_fuel_data_count) 

            # print('data_generation_count: ', data_generation_count)

            #points generation loop
            for j in range(selected_fuel_data_count):
                data_point = pd.Series(specific_fuel_dataset.loc[j])   #data point selected to generate more data points

                #calling fucntion
                generated_data_frame = generate_data_points.data_point_generator(data_point,data_generation_count,data)
                # print('generated_data_frame: ', generated_data_frame)
                
                #appending result 
                extended_dataframe = extended_dataframe.append(generated_data_frame)    #appending to final dataframe

        #writing file 
        extended_dataframe.to_csv(str(Main_folder_dir)+'/data/extended_data.csv')
        return extended_dataframe

    def data_point_generator(data_point,data_generation_count,data_frame):
        '''
        generating more data points using,
        // np.random.multivariate_normal(mean, cov, 2000).T //
        #REf : http://onlinestatbook.com/2/estimation/mean.html
        #REF : https://datascienceplus.com/understanding-the-covariance-matrix/
        Note that the standard deviation of a sampling distribution is its standard error.
        '''
        #taking header from data_frame
        generated_data_frame = data_frame[0:0]
        # print('generated_data_frame: ', generated_data_frame)
        generated_data_frame = generated_data_frame.append(data_point)
        # print('generated_data_frame: ', generated_data_frame)
        generated_data_frame =  pd.concat([generated_data_frame]*data_generation_count, ignore_index=True)
        # print('generated_data_frame: ', generated_data_frame)
        T_mean = data_point['T(K)']
        T_SD = data_point['T_Error(%)']/100  #1%
        # T_upperbound = T_mean + 1.96* (T_mean/100)
        # T_lowerbound = T_mean - 1.96* (T_mean/100)

        P_mean = data_point['P(atm)']
        P_SD = data_point['P_Error(%)']/100
        # P_upperbound = P_mean + 1.96* (P_mean/100)
        # P_lowerbound = P_mean - 1.96* (P_mean/100)

        tau_mean = data_point['Time(μs)']
        tau_SD =  20/100          

        mean = [T_mean, P_mean,tau_mean]
        cov = [[T_SD**2, 0,0], [0, P_SD**2,0],[0, 0,tau_SD**2]]  # diagonal covariance

        ####module from samapling out of 2000 points
        T_gen, P_gen, tau_gen = np.random.multivariate_normal(mean, cov, 2000).T    
        data_point_generated = list(zip(T_gen,P_gen,tau_gen))

        #picking up random points 
        sampling = random.choices(data_point_generated, k=data_generation_count)

        # import matplotlib.pyplot as plt
        # T_gen, P_gen, tau_gen = np.random.multivariate_normal(mean, cov, data_generation_count).T    
        # data_point_generated = list(zip(T_gen,P_gen,tau_gen))

        # #picking up random points 
        # import random 
        # sampling = data_point_generated

        for i,item in enumerate(sampling):
            # print(sampling[i])
            generated_data_frame['T(K)'].loc[i] = sampling[i][0]
            generated_data_frame['P(atm)'].loc[i] = sampling[i][1]
            generated_data_frame['Time(μs)'].loc[i] = sampling[i][2]

      

        # fig = plt.figure()
        # ax = plt.axes(projection="3d")
        # ax.scatter3D(T_gen,P_gen,tau_gen,c="pink")
        # ax.scatter3D(generated_data_frame['T(K)'],generated_data_frame['P(atm)'],generated_data_frame['Time(μs)'], c="black")
        # ax.set_xlabel('Temperature')
        # ax.set_ylabel('Pressure')
        # ax.set_zlabel('Ignition Delay')
        # plt.show()

        return generated_data_frame