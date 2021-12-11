import pandas as pd 
import numpy as np
import warnings

class select_feature():
	'''
	common framework to call generate and transform independent feature and 
	'''

	def feature_selection(dataset):    		
            #constructing feature for Dataset
            df =pd.DataFrame([])
            # X_names = ['log_Temp(K)','Temp(K)','log_P(atm)','log_Fuel(%)','log_Oxidizer(%)','log_Diluent(%)','log_Equv(%)',
            # 'C1', 'C2', 'C3', 'C4','P_P', 'P_S','P_T','P_Q', 'S_S', 'S_T', 'S_Q', 'T_T', 'T_Q',
            # 'Q_Q', 'P_H', 'S_H', 'T_H']

            #Obtaining required column from the dataset
            # df['Primary_C'] = (dataset['Primary_C'])  # 0
            # df['Secondary_C'] = (dataset['Secondary_C'])  # 1
            # df['Tertiary_C'] = (dataset['Tertiary_C'])  # 2
            # df['Quaternary_C'] = (dataset['Quaternary_C'])  # 3 
            # df['log_Temp(K)'] = np.log(dataset['T(K)'])  # 4  #0
            T_0 = 1000
            P_0 = 1
            # df['SH'] = dataset['S_H']  # 9  #5            
            # df['SH_log_Temp(K)'] = dataset['S_H'] * np.log(dataset['T(K)'] /T_0 )  # 5  #1
            # df['SH_log_P(atm)'] = dataset['S_H'] * np.log(dataset['P(atm)']/P_0)  # 6  #2
            # df['SH_log_Fuel(%)'] = dataset['S_H'] * np.log(dataset['Fuel(%)']/5)  # 7  #3
            # df['SH_log_Oxidizer(%)'] = dataset['S_H'] * np.log(dataset['Oxidizer(%)']/20)  # 8  #4
            # df['SH_log_Diluent(%)'] = dataset['S_H'] * np.log(dataset['Diluant(%)']/10)  # 9  #5      
            # 
            # 
            
            # df['log_Temp(K)'] = np.log(dataset['T(K)'] ) / T_0  # 5  #1
            df['log_P(atm)'] = np.log(dataset['P(atm)']) / P_0 # 6  #2
            # df['PT'] = (np.log(dataset['T(K)'] ) / T_0)*(np.log(dataset['P(atm)']) / P_0)
            df['log_Fuel(%)'] = np.log(dataset['Fuel(%)'])  # 7  #3
            # df['log_S_H'] = dataset['S_H'] / dataset['T(K)']# 26 #18
            df['log_Oxidizer(%)'] = np.log(dataset['Oxidizer(%)'])  # 8  #4
            # df['log_Diluent(%)'] = np.log(dataset['Diluant(%)'])  # 9  #5
            df['T0/S_H__T'] = T_0/(dataset['S_H'] * dataset['T(K)'])# 26 #18
            df['T0/T'] = T_0/dataset['T(K)']
            # df['log_T0/T'] = np.log(dataset['T(K)']/T_0)
            # df['1/T'] = 1/dataset['T(K)']     
            # df['E_0'] = np.ones(dataset.shape[0]) / dataset['T(K)']     
            # df['P_S'] = dataset['P_S']   # 16 #8 
            # df['P_H'] = dataset['P_H']   # 25 #17
            # df['S_S'] = dataset['S_S']  # 19 #11
            

            ################################
            # Adding constant Term in front#
            ################################
            #adding 'constant' name in the column headers
            df.insert(0, 'Constant', np.ones(df.shape[0]))

            #################################
            # changing nan values with zeros#
            #################################
            # '''
            # Just to make sure that dataset contains no NAN and Zeros
            # '''
            #np.sum(df.isnull().sum()) 
            #if any cell is null then generating rowwise sum and then adding sum all rows 
            print('\n')
            if(np.sum(df.isnull().sum()) != 0):                
                warnings.warn('\n Dataset contains some empty cell or NAN values')
                df = df.fillna(0)   

            print('\n')
            if(np.sum(df.any().sum) != 0):
                warnings.warn('\n Column containing all zeros are removed')
                df = df.loc[:, df.any()]             #Removing All the columns with zeros entries 

            #how many rows there are with "one or more NaNs"
            No_of_NaN = df.isnull().T.any().T.sum()
            print('\n Total NaN values in data: ', No_of_NaN)

            if(No_of_NaN != 0):
                nan_rows = df[df.isnull().T.any().T]
                print('Dataframe with NaN values are:\n')

            #independent feature
            tau = np.log(dataset['Time(μs)']) ###IT'S log time
            return df,tau 
        
	def column_selection():
		'''
		For result output column headers will be selected from these features
		'''
		columns = ['Constant','log_P(atm)','log_Fuel(%)','log_Oxidizer(%)','T0/S_H__T','T0/T']
		return columns

	def bond_extraction_cols():
		'''
		columns for bond selection
		'''
		columns = ['Primary_C', 'Secondary_C', 'Tertiary_C', 'Quaternary_C', 'Other_Atom', 'P_P', 'P_S', 'P_T',
		'P_Q', 'S_S', 'S_T', 'S_Q', 'T_T', 'T_Q', 'Q_Q', 'P_H', 'S_H', 'T_H']
		return columns


        
