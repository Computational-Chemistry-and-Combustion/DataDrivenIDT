import pandas as pd
import numpy as np
import os 
def seperate(uniqueFuel):
        cols = ['Diluant(%)','Equv(phi)','Fuel','Fuel(%)','Oxidizer(%)','P(atm)','T(K)','Time(μs)']
        c0 = pd.read_csv('cluster_0.csv')
        c1 = pd.read_csv('cluster_1.csv')
        c2 = pd.read_csv('cluster_2.csv')
        c0 = c0[cols]
        c1 = c1[cols]
        c2 = c2[cols]
        for i in uniqueFuel:
                os.mkdir('./SperatedFuels/'+str(i))
                c0[c0['Fuel']==i].to_csv('./SperatedFuels/'+str(i)+'/cluster0.csv')
                c1[c1['Fuel']==i].to_csv('./SperatedFuels/'+str(i)+'/cluster1.csv')
                c2[c2['Fuel']==i].to_csv('./SperatedFuels/'+str(i)+'/cluster2.csv')


c0 = pd.read_csv('cluster_0.csv')
c1 = pd.read_csv('cluster_1.csv')
c1 = pd.read_csv('cluster_2.csv')

# one data set is enough 
uniqueFuel = c0['Fuel'].unique()
seperate(uniqueFuel)
print('uniqueFuel: ', uniqueFuel)