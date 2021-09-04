import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

fName = ['Cluster-1','Cluster-2','Cluster-3']
folder = 'Analysis_clusters_3'
path = './'+folder+'/result/cluster_data_before_regression/'
files = os.listdir(path)
fileName = [k.split('.')[0] for k in files]
data = pd.read_csv('./data/full_data.csv')
uniqueFuel = sorted(data['Fuel'].unique())
try:
    os.mkdir('./IMGresult/Temp/'+str(folder)+'/')
except FileExistsError:
    pass
for i in uniqueFuel:
    dataTaken = False
    FuelWiseDataFrame = pd.DataFrame([])
    for j in fileName:
        if dataTaken == True:
            AllDataFuel_j = data[data['Fuel']==i]['T(K)'] #total number of fuel-j in main dataset
            AllDataFuel_j = AllDataFuel_j.reset_index(drop=True)
            FuelWiseDataFrame.loc[:,'all'] = AllDataFuel_j
            dataTaken = True
        # print('\n')
        print(fileName)
        clusterData = pd.read_csv(path+j+'.csv')
        clusterData = clusterData.loc[:, ~clusterData.columns.str.contains('^Unnamed')]
        DataFuel_j = clusterData[clusterData['Fuel']==i]['T(K)'] #count of fuel-j in cluster-i
        DataFuel_j= DataFuel_j.reset_index(drop=True)
        print('Hi')
        print(DataFuel_j)
        FuelWiseDataFrame.loc[:,j] = DataFuel_j
        
    print(FuelWiseDataFrame)
    plt.rc('text', usetex=True)
    fontsize=19
    FuelWiseDataFrame.plot.hist(bins=10, alpha=0.2)
    plt.xlabel("Temp")
    plt.ylabel("freq")
    plt.title(i)
    plt.savefig('./IMGresult/Temp/'+str(folder)+'/'+i+'.eps',dpi=600,orientation ='landscape')
    plt.close()

  


