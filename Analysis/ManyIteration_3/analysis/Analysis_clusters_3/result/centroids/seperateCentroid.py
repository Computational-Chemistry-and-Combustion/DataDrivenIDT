import pandas as pd
import numpy as no

def sperated(cen):
        df0 = pd.read_csv('centroid_0.csv')
        df1 = pd.read_csv('centroid_1.csv')
        df2 = pd.read_csv('centroid_2.csv')
        
        data = pd.DataFrame([],columns=df0.columns)

        for i in range(len(df0)): #as 5000 iteration
                if(cen['0'][i] == 0):
                        data = data.append(df0.iloc[i,:])
                if(cen['0'][i] == 1):
                        data = data.append(df1.iloc[i,:])
                if(cen['0'][i] == 2):
                        data = data.append(df2.iloc[i,:])                       

        return data

c0 = pd.read_csv('c0.csv')
high = sperated(c0) 
high.to_csv('High.csv')

c1 = pd.read_csv('c1.csv')
mid = sperated(c1) 
mid.to_csv('mid.csv')

c2 = pd.read_csv('c2.csv')
low = sperated(c2) 
low.to_csv('low.csv')




