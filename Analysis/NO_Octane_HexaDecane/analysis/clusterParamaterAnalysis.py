import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

no_folders = 7
for x in range(1,no_folders):
    folder = 'Analysis_clusters_'+str(x)
    path = './'+folder+'/result/cluster_data_before_regression/'
    files = os.listdir(path)
    fileName = [k.split('.')[0] for k in files]
    data = pd.read_csv('./data/full_data.csv')
    properties = ['Equv(phi)','Fuel(%)','Oxidizer(%)','P(atm)','T(K)','Time(μs)']

    # path2 = './'+folder+'/result/final_cluster/'
    # print(path2)
    # files2 = os.listdir(path2)
    # fileName2 = [k.split('.')[0] for k in files2]
    # print(fileName2)
    # final_prop = ['log_P(atm)','log_Fuel(%)','log_Oxidizer(%)','T0/S_H__T','T0/T']
    # final_propName = ['log_P(atm)','log_Fuel(%)','log_Oxidizer(%)','T0_S_H__T','T0_T']

    # try:
    #     os.mkdir('./IMGresult/param/'+str(x)+'/')
    # except FileExistsError:
    #     pass
    # for i in fileName:
    #     dataTaken = False
    #     ClusterWiseData = pd.DataFrame([])
    #     clusterData = pd.read_csv(path+i+'.csv')
    #     for j in properties:
    #         clusterData = clusterData.loc[:, ~clusterData.columns.str.contains('^Unnamed')]
    #         DataFuel_j = clusterData[j] #count of fuel-j in cluster-i
    #         ax = sns.distplot(DataFuel_j, hist=True, kde=False, color = 'blue',hist_kws={'edgecolor':'black'},bins=10)
    #         try:
    #             os.mkdir('./IMGresult/param/'+str(x)+'/'+str(i)+'/')
    #         except FileExistsError:
    #             pass
    #         plt.savefig('./IMGresult/param/'+str(x)+'/'+str(i)+'/'+j+'.jpg')
    #         plt.close()
    
    # for i in fileName2:
    #     dataTaken = False
    #     ClusterWiseData = pd.DataFrame([])
    #     clusterData = pd.read_csv(path2+i+'.csv')
    #     count = 0
    #     for j in final_prop:
    #         clusterData = clusterData.loc[:, ~clusterData.columns.str.contains('^Unnamed')]
    #         DataFuel_j = clusterData[j] #count of fuel-j in cluster-i
    #         ax = sns.distplot(DataFuel_j, hist=True, kde=False, color = 'blue',hist_kws={'edgecolor':'black'},bins=10)
    #         try:
    #             os.mkdir('./IMGresult/param/'+str(x)+'/'+str(i)+'/')
    #         except FileExistsError:
    #             pass
    #         plt.savefig('./IMGresult/param/'+str(x)+'/'+str(i)+'/'+final_propName[count]+'.jpg')
    #         plt.close()
    #         count = count + 1

    for i in properties:
        ClusterWiseData = pd.DataFrame([])
        for j in fileName:
            clusterData = pd.read_csv(path+j+'.csv')
            clusterData = clusterData.loc[:, ~clusterData.columns.str.contains('^Unnamed')]
            ClusterWiseData[j] = clusterData[i] #count of fuel-j in cluster-i
            
        #normal plt
        N = len(fileName)
        ind = np.arange(N)
        width = 0.1
        color = ['r','b','g','c','y','m','k','w']
        bar = []
        count = 0
        print(ClusterWiseData)
        for k in fileName:
            vals = ClusterWiseData[k].to_numpy()
            print(vals)
            bar.append(plt.bar(ind+(width*count), vals, width, color=color[count]))
            count = count +1
            
        plt.xlabel(i)
        plt.ylabel("freq")
        plt.title(i)
        plt.xticks(ind+width)
        plt.legend( tuple(bar) , fileName )
        try:
            os.mkdir('./IMGresult/ParaCombined/'+str(x)+'/')
        except FileExistsError:
            pass
        plt.savefig('./IMGresult/ParaCombined/'+str(x)+'/'+str(i)+'.jpg')
        plt.close()

  


