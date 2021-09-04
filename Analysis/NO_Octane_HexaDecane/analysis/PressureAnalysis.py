import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

no_folders = 9
for x in range(1,no_folders):
    folder = 'Analysis_clusters_'+str(x)
    path = './'+folder+'/result/cluster_data_before_regression/'
    files = os.listdir(path)
    fileName = [k.split('.')[0] for k in files]
    data = pd.read_csv('./data/full_data.csv')
    uniqueFuel = sorted(data['Fuel'].unique())
    try:
        os.mkdir('./IMGresult/Pressure/'+str(x)+'/')
    except FileExistsError:
        pass
    for i in uniqueFuel:
        dataTaken = False
        FuelWiseDataFrame = pd.DataFrame([])
        for j in fileName:
            if dataTaken == False:
                AllDataFuel_j = data[data['Fuel']==i]['P(atm)'] #total number of fuel-j in main dataset
                AllDataFuel_j = AllDataFuel_j.reset_index(drop=True)
                FuelWiseDataFrame.loc[:,'all'] = AllDataFuel_j
                dataTaken = True
            # print('\n')
            print(fileName)
            clusterData = pd.read_csv(path+j+'.csv')
            clusterData = clusterData.loc[:, ~clusterData.columns.str.contains('^Unnamed')]
            DataFuel_j = clusterData[clusterData['Fuel']==i]['P(atm)'] #count of fuel-j in cluster-i
            DataFuel_j= DataFuel_j.reset_index(drop=True)
            print('Hi')
            print(DataFuel_j)
            FuelWiseDataFrame.loc[:,j] = DataFuel_j
            
        print(FuelWiseDataFrame)
        FuelWiseDataFrame.plot.hist(bins=10, alpha=0.2)
        plt.xlabel("Temp")
        plt.ylabel("freq")
        plt.title(i)
        plt.savefig('./IMGresult/Pressure/'+str(x)+'/'+i+'.jpg')
        plt.close()

  


