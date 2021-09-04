import pandas as pd
import numpy as no

def sperated():
        df0 = pd.read_csv('Result_Coefficients_0.csv')
        df1 = pd.read_csv('Result_Coefficients_1.csv')
        
        high = pd.DataFrame([],columns=df0.columns)
        low = pd.DataFrame([],columns=df0.columns)
        for i in range(5000): #as 5000 iteration
                if(df0['training_adj_r2'][i] > df1['training_adj_r2'][i]):
                        # print('zero high')
                        # print(df0.iloc[i,:])
                        high = high.append(df0.iloc[i,:])
                        low=low.append(df1.iloc[i,:])
                else:
                        high=high.append(df1.iloc[i,:])
                        low=low.append(df0.iloc[i,:])

                
        high.to_csv('high.csv')
        low.to_csv('low.csv')

sperated()            





