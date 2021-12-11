import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def tempAnalysis(curr_dir,dataset_location):
    path = curr_dir+'/result/cluster_data_before_regression/'
    files = os.listdir(path)
    fileName = [k.split('.')[0] for k in files]
    # fName = ['Cluster-'+str(x) for x in range(len(files))]
    data = pd.read_csv(dataset_location)
    uniqueFuel = sorted(data['Fuel'].unique())
    try:
        os.mkdir(curr_dir+'/result/AnalysisResult/Temperature/')
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
            clusterData = pd.read_csv(path+j+'.csv')
            clusterData = clusterData.loc[:, ~clusterData.columns.str.contains('^Unnamed')]
            DataFuel_j = clusterData[clusterData['Fuel']==i]['T(K)'] #count of fuel-j in cluster-i
            DataFuel_j= DataFuel_j.reset_index(drop=True)
            FuelWiseDataFrame.loc[:,j] = DataFuel_j
            
        print(FuelWiseDataFrame)
        # plt.rc('text', usetex=True)
        FuelWiseDataFrame.plot.hist(bins=10, alpha=0.3)
        fontsize=15
        plt.xlabel("Temperature (K)",fontsize=fontsize)
        plt.ylabel(" Count of data points in clusters ",fontsize=fontsize) 
        plt.title("Temperature vs number of data points",fontsize=fontsize)
        # plt.rc('text', usetex=True)
        # plt.legend( fName ,fontsize=fontsize-5,loc='best')
        plt.tight_layout()
        plt.savefig(curr_dir+'/result/AnalysisResult/Temperature/'+i+'.jpg',dpi=600)
        plt.close()

    


