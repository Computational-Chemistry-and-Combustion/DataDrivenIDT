import pandas as pd 
import numpy as np
import warnings
import time
class select_feature():
	'''
	common framework to call generate and transform independent feature and 
	'''

	def feature_selection(dataset):    		
            #constructing feature for Dataset
            df =pd.DataFrame([])

            df = dataset.iloc[:,:-1] # 6  #2
            df['volatile_acidity'] = df['volatile_acidity'] * 10
            df['citric_acid'] = df['citric_acid'] * 10
            df['chlorides'] = df['chlorides'] * 100
            df['free_sulfur_dioxide'] = df['free_sulfur_dioxide'] / 10
            df['total_sulfur_dioxide'] = df['total_sulfur_dioxide'] / 100
            df['density'] = df['density'] * 10
            df['sulphates'] = df['sulphates'] * 10

        
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
            tau = dataset.iloc[:,-1] ###IT'S log time
            return df,tau 
        
	def column_selection():
		'''
		For result output column headers will be selected from these features
		'''     
		columns = ['Constant','fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol']
		return columns

	def bond_extraction_cols():
		'''
		columns for bond selection
		'''
		columns = ['Primary_C', 'Secondary_C', 'Tertiary_C', 'Quaternary_C', 'Other_Atom', 'P_P', 'P_S', 'P_T',
		'P_Q', 'S_S', 'S_T', 'S_Q', 'T_T', 'T_Q', 'Q_Q', 'P_H', 'S_H', 'T_H']
		return columns


        
