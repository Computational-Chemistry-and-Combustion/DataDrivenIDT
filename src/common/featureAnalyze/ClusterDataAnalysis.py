import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def ClusterDataAnalysis(curr_dir,dataset_location):
    path = curr_dir+'/result/cluster_data_before_regression/'
    files = os.listdir(path)
    fileName = [i.split('.')[0] for i in files]
    fName = ['Cluster-'+str(x) for x in range(len(files))]
    data = pd.read_csv(dataset_location)
    uniqueFuel = sorted(data['Fuel'].unique())
    clusterFuelCount = []
    clusterFuelCountFraction = []
    for i in files:
        FuelCounts = []
        FractionCount = []
        clusterData = pd.read_csv(path+str(i))
        for j in uniqueFuel:
            countOfFuel_j = len(clusterData[clusterData['Fuel']==j]==True) #count of fuel-j in cluster-i
            FuelCounts.append(countOfFuel_j)
            TotalCountOfFuel_j = len(data[data['Fuel']==j]==True) #total number of fuel-j in main dataset
            # print('\n')
            # print(TotalCountOfFuel_j)
            FractionCount.append(countOfFuel_j/TotalCountOfFuel_j) #calculation fraction
        
        clusterFuelCount.append(FuelCounts)
        clusterFuelCountFraction.append(FractionCount)
    # print(uniqueFuel)
    # print(clusterFuelCount)
    # print(clusterFuelCountFraction)
    fuelLength = [len(i) for i in uniqueFuel]

    #normal plto
    N = len(fuelLength)
    ind = np.arange(N) 
    width = 0.1
    color = ['r','b','g','c','y','m','k','w']
    bar = []
    for i in range(len(fileName)):
        vals = clusterFuelCount[i][:]
        bar.append(plt.bar(ind+(width*i), vals, width, color=color[i]))

    plt.rc('text', usetex=True)
    fontsize=19
    plt.xlabel("Length of straight chain alkane",fontsize=fontsize)
    plt.ylabel("Count of data points",fontsize=fontsize)
    plt.title("Count of fuel in each cluster",fontsize=fontsize)
    plt.xticks(ind+width,fuelLength,fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend( tuple(bar) , fName ,fontsize=fontsize-5,loc='best')
    plt.tight_layout()
    try:
        plt.savefig(curr_dir+'/result/AnalysisResult/fuelDataCount/dataFuel.eps',dpi=600,orientation ='landscape')
    except FileNotFoundError:
        os.mkdir(curr_dir+'/result/AnalysisResult/fuelDataCount/')
        plt.savefig(curr_dir+'/result/AnalysisResult/fuelDataCount/dataFuel.eps',dpi=600,orientation ='landscape')
    plt.show()

    #fraction plot
    #normal plto
    bar = []
    for i in range(len(fileName)):
        vals = clusterFuelCountFraction[i][:]
        bar.append(plt.bar(ind+(width*i), vals, width, color=color[i]))
        
    plt.xlabel("Length of straight chain alkane",fontsize=fontsize)
    plt.ylabel("Fraction of data points",fontsize=fontsize)
    plt.title("Fraction of fuel in each cluster",fontsize=fontsize)
    plt.xticks(ind+width,fuelLength,fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend( tuple(bar) , fName ,fontsize=fontsize-5,loc='best')
    plt.tight_layout()
    try:
        plt.savefig(curr_dir+'/result/AnalysisResult/fuelDataCount/dataFuel_fraction.eps',dpi=600,orientation ='landscape')
    except FileNotFoundError:
        os.mkdir(curr_dir+'/result/AnalysisResult/fuelDataCount/')
        plt.savefig(curr_dir+'/result/AnalysisResult/fuelDataCount/dataFuel_fraction.eps',dpi=600,orientation ='landscape')
    plt.show()




