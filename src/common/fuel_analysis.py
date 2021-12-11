import pandas as pd
import copy
import matplotlib.pyplot as plt
import shutil
import os

class fuel_analysis():
    def makedir(dirs,curr_directory):
        '''
        it generates the path 
        '''
        # define the name of the directory to be created
        path = str(curr_directory)+"/result/Fuel_Parameter_Histogram/"+str(dirs)+'/'
        os.makedirs(path)
        return path
    
    def makedir_cluster(dirs,cluster_num,curr_directory):
        '''
        it generates the path 
        '''
        # define the name of the directory to be created
        path = str(curr_directory)+"/result/seperated_clusters/cluster"+str(cluster_num)+'/'+str(dirs)+'/'
        os.makedirs(path)
        return path

    def find_strightchain_alkanes_dataset(Fuel_Name_data):
            '''
            This module will find only stright chain alkanes from the dataset
            and it 
            '''
            unique_fuels = list(Fuel_Name_data['Fuel'].unique())   #findind out all unique fuels               
            list_fuel = []  #List of fuels to store
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
            
            Data = copy.deepcopy(Fuel_Name_data)
            dataset = pd.DataFrame([])
            for i,item in enumerate(list_fuel):
                    # print('Data.Fuel == list_fuel[i]: ', Data.Fuel == list_fuel[i])
                    dataset = dataset.append(Data[Data.Fuel == list_fuel[i]]) #filetring dataset according list fuels
            dataset = dataset.reset_index(drop=True)        
            
            return dataset,list_fuel

    def replace_nan_with_least(data):
        '''
        repalces least value with least value in the column
        '''
        least_temp_error = data['T_Error(%)'].dropna().min(skipna=True)
        data['T_Error(%)'] = data['T_Error(%)'].fillna(least_temp_error)
        data['T_Error(%)'] = data['T_Error(%)'].str.strip('')
        least_press_error = data['P_Error(%)'].dropna().min(skipna=True)
        data['P_Error(%)'] = data['P_Error(%)'].fillna(least_temp_error)
        data['P_Error(%)'] = data['P_Error(%)'].str.strip('')
        return data


    def RemovePlusMinus(data):
        '''
        removes plus_minus symbol
        '''
        # print(data['T_Error(%)'])
        data['T_Error(%)'] = data['T_Error(%)'].replace('±','',regex=True)
        data['P_Error(%)'] = data['P_Error(%)'].replace('±','',regex=True)
        return data
        
    def fuel_data_analysis(data,curr_directory):
        '''
        It seperates all fuel and fuelwise gives upper and lower bound of parameter of each fuel 
        Also
        Stores histogram of parameters by generaing directory  
        '''
        data = fuel_analysis.RemovePlusMinus(data)
        data = fuel_analysis.replace_nan_with_least(data)
        # data.to_csv('check.csv')
        #find alkanes
        alkanes_data,uniq_fuel = fuel_analysis.find_strightchain_alkanes_dataset(data)
        if(os.path.isdir(str(curr_directory+'/result/Fuel_Parameter_Histogram'))):
            shutil.rmtree(str(curr_directory+'/result/Fuel_Parameter_Histogram'))
            os.makedirs(str(curr_directory+'/result/Fuel_Parameter_Histogram'))
        else:
            os.makedirs(str(curr_directory+'/result/Fuel_Parameter_Histogram'))

        #printing data range its histogram
        for i,item in enumerate(uniq_fuel):
                    curr_dir = fuel_analysis.makedir(uniq_fuel[i],curr_directory)
                    #removing pm symbol from 
                    specific_fuel = alkanes_data[alkanes_data.Fuel == uniq_fuel[i]]  #filetring dataset according list fuels
                    print('\n')
                    print('Data of Fuel  :',uniq_fuel[i])
                    #TEMPERATURE FRQ
                    plt.figure(10*i+0)
                    plt.rc('text', usetex=True)
                    fontsize=19 
                    plt.hist(specific_fuel['T(K)'])
                    plt.ylabel('Frequency',fontsize=fontsize)
                    plt.xlabel('T(K)',fontsize=fontsize)
                    plt.savefig(str(curr_dir)+str(uniq_fuel[i])+'_temp.eps')
                    print('Maximum Temeprature : ',max(specific_fuel['T(K)']))
                    print('Minimum Temeprature : ',min(specific_fuel['T(K)']))
                    #TEMPERATURE uncertain
                    # plt.figure(10*i+5)
                    # plt.hist(specific_fuel['T_Error(%)'])
                    # plt.savefig(str(curr_dir)+str(uniq_fuel[i])+'_temp_err.png')
                    print('Maximum Temeprature Error : ',max(specific_fuel['T_Error(%)']))
                    print('Minimum Temeprature Error : ',min(specific_fuel['T_Error(%)']))
                    
                    #pressure
                    plt.figure(10*i+1)
                    plt.hist(specific_fuel['P(atm)'])
                    plt.ylabel('Frequency',fontsize=fontsize)
                    plt.xlabel('P(atm)',fontsize=fontsize)
                    plt.savefig(str(curr_dir)+str(uniq_fuel[i])+'_press.eps')
                    print('Maximum Pressure : ',max(specific_fuel['P(atm)']))
                    print('Minimum Pressure : ',min(specific_fuel['P(atm)']))
                    #Pressure uncertain
                    # plt.figure(10*i+0)
                    # plt.hist(specific_fuel['P_Error(%)'])
                    # plt.savefig(str(curr_dir)+str(uniq_fuel[i])+'_pressure_err.png')
                    print('Maximum Pressure Error: ',max(specific_fuel['P_Error(%)']))
                    print('Minimum Pressure Error: ',min(specific_fuel['P_Error(%)']))
                    
                    #Fuel
                    plt.figure(10*i+2)        
                    plt.hist(specific_fuel['Fuel(%)'])
                    plt.ylabel('Frequency',fontsize=fontsize)
                    plt.xlabel('Fuel($\%$)',fontsize=fontsize)
                    plt.savefig(str(curr_dir)+str(uniq_fuel[i])+'_fuel.eps')

                    print('Maximum Fuel% : ',max(specific_fuel['Fuel(%)']))
                    print('Minimum Fuel% :',min(specific_fuel['Fuel(%)']))

                    #oxidizer
                    plt.figure(10*i+3)
                    plt.hist(specific_fuel['Oxidizer(%)'])
                    plt.ylabel('Frequency',fontsize=fontsize)
                    plt.xlabel('Oxidizer($\%$)',fontsize=fontsize)                    
                    plt.savefig(str(curr_dir)+str(uniq_fuel[i])+'_oxi.eps')
                    print('Maximum Oxidizer% : ',max(specific_fuel['Oxidizer(%)']))
                    print('Minimum Oxidizer% : ',min(specific_fuel['Oxidizer(%)']))

                    #group
                    plt.hist(specific_fuel['Research_group'])

                    #Equivalecne Ratio
                    plt.figure(10*i+4)
                    plt.hist(specific_fuel['Equv(phi)'])
                    plt.ylabel('Frequency',fontsize=fontsize)
                    plt.xlabel('Equv(phi)',fontsize=fontsize)                    
                    plt.savefig(str(curr_dir)+str(uniq_fuel[i])+'_equi.eps')
                
                    print('Equv(phi): ',max(specific_fuel['Equv(phi)']))
                    print('Equv(phi) : ',min(specific_fuel['Equv(phi)']))
                    plt.close("all")

                    print('Number of Datapoints:',len(specific_fuel))
        
        print('\n\n\n\To check Histograms result go to the Directory: ./result/Fuel_Parameter_Histogram  \n\n\n')

                    

if __name__ == "__main__":                     
    #read data
    data  = pd.read_csv(r'Alkane_Dataset_3.csv')
    path = '/home/pragnesh/Git/Data_driven_Kinetics/CleanCode/testing_place'
    fuel_analysis.fuel_data_analysis(data,path)