import pandas as pd
import numpy as no

def sperated():
        df0 = pd.read_csv('Result_Coefficients_0.csv')
        df1 = pd.read_csv('Result_Coefficients_1.csv')
        df2 = pd.read_csv('Result_Coefficients_2.csv')
        
        high = pd.DataFrame([],columns=df0.columns)
        mid = pd.DataFrame([],columns=df0.columns)
        low = pd.DataFrame([],columns=df0.columns)

        # will help t identify cluster and its index
        centroid_0 = pd.DataFrame([])
        centroid_1 = pd.DataFrame([])
        centroid_2 = pd.DataFrame([])
        for i in range(5000): #as 5000 iteration
                if(df0['training_adj_r2'][i] > df1['training_adj_r2'][i] and df0['training_adj_r2'][i] > df2['training_adj_r2'][i]):
                        # print('zero high')
                        # print(df0.iloc[i,:])
                        high = high.append(df0.iloc[i,:])
                        centroid_0 = centroid_0.append([0])
                        if(df1['training_adj_r2'][i] > df2['training_adj_r2'][i]):
                                mid=mid.append(df1.iloc[i,:])
                                centroid_1 = centroid_1.append([1])

                                low=low.append(df2.iloc[i,:])
                                centroid_2 = centroid_2.append([2])
                        else:
                                mid=mid.append(df2.iloc[i,:])
                                centroid_1 = centroid_1.append([2])

                                low=low.append(df1.iloc[i,:])
                                centroid_2 = centroid_2.append([1])

                elif(df1['training_adj_r2'][i] > df0['training_adj_r2'][i] and df1['training_adj_r2'][i] > df2['training_adj_r2'][i]):
                        # print('first high')
                        # print(df1.iloc[i,:])
                        high=high.append(df1.iloc[i,:])
                        centroid_0 = centroid_0.append([1])
                        if(df0['training_adj_r2'][i] > df2['training_adj_r2'][i]):
                                mid=mid.append(df0.iloc[i,:])
                                centroid_1 = centroid_1.append([0])

                                low=low.append(df2.iloc[i,:])
                                centroid_2 = centroid_2.append([2])
                        else:
                                mid=mid.append(df2.iloc[i,:])
                                centroid_1 = centroid_1.append([2])

                                low=low.append(df0.iloc[i,:])
                                centroid_2 = centroid_2.append([0])                
                else:
                        # print('second High')
                        # print(df2.iloc[i,:])
                        high=high.append(df2.iloc[i,:])
                        centroid_0 = centroid_0.append([2])
                        if(df1['training_adj_r2'][i] > df0['training_adj_r2'][i]):
                                mid=mid.append(df1.iloc[i,:])
                                centroid_1 = centroid_1.append([1])

                                low=low.append(df0.iloc[i,:])                                
                                centroid_2 = centroid_2.append([0])
                        else:
                                mid=mid.append(df0.iloc[i,:])
                                centroid_1 = centroid_1.append([0])

                                low=low.append(df1.iloc[i,:])
                                centroid_2 = centroid_2.append([1])

        # high.to_csv('high.csv')
        # mid.to_csv('mid.csv')
        # low.to_csv('low.csv')

        centroid_0.to_csv('c0.csv')
        centroid_1.to_csv('c1.csv')
        centroid_2.to_csv('c2.csv')

sperated()            





