import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

no_folders = 9
for x in range(1,no_folders):
    folder = 'Analysis_clusters_'+str(x)
    path = './'+folder+'/result/cluster_data_before_regression/'
    files = os.listdir(path)
    fileName = [i.split('.')[0] for i in files]
    data = pd.read_csv('./data/full_data.csv')
    uniqueFuel = sorted(data['Fuel'].unique())
    clusterFuelCount = []
    clusterFuelCountFraction = []
    for i in files:
        FuelProp = []
        clusterData = pd.read_csv(path+str(i))
        print(clusterData)
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
    print(fuelLength)

    #normal plto
    N = len(fuelLength)
    ind = np.arange(N) 
    width = 0.1
    color = ['r','b','g','c','y','m','k','w']
    bar = []
    for i in range(len(fileName)):
        vals = clusterFuelCount[i][:]
        bar.append(plt.bar(ind+(width*i), vals, width, color=color[i]))
        
    plt.xlabel("length of staright chain alkane")
    plt.ylabel("Count of data poins")
    plt.title("Count of fuel in each cluster")
    plt.xticks(ind+width,fuelLength)
    plt.legend( tuple(bar) , fileName )
    plt.savefig('./IMGresult/'+folder+'.jpg')
    plt.show()

    #fraction plot
    #normal plto
    bar = []
    for i in range(len(fileName)):
        vals = clusterFuelCountFraction[i][:]
        bar.append(plt.bar(ind+(width*i), vals, width, color=color[i]))
        
    plt.xlabel("length of staright chain alkane")
    plt.ylabel("Fraction of data poins")
    plt.title("Fraction of fuel in each cluster")
    plt.xticks(ind+width,fuelLength)
    plt.legend( tuple(bar) , fileName )
    plt.savefig('./IMGresult/'+folder+'_fraction.jpg')
    plt.show()




