import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

fName = ['Cluster-1','Cluster-2','Cluster-3']
folder = 'Analysis_clusters_3'
path = './'+folder+'/result/cluster_data_before_regression/'
files = os.listdir(path)
fileName = [k.split('.')[0] for k in files]
data = pd.read_csv('./data/full_data.csv')
print(data.columns)
properties = ['Equv(phi)','Fuel(%)','Oxidizer(%)','P(atm)','T(K)','Time(μs)']

path2 = './'+folder+'/result/final_cluster/'
print(path2)
files2 = os.listdir(path2)
fileName2 = [k.split('.')[0] for k in files2]
print(fileName2)
final_prop = ['log_P(atm)','log_Fuel(%)','log_Oxidizer(%)','T0/S_H__T','T0/T']
final_propName = ['log_P(atm)','log_Fuel(%)','log_Oxidizer(%)','T0_S_H__T','T0_T']

try:
    os.mkdir('./IMGresult/param/'+str(folder)+'/')
except FileExistsError:
    pass

#prop plots
for i in fileName:
    dataTaken = False
    ClusterWiseData = pd.DataFrame([])
    clusterData = pd.read_csv(path+i+'.csv')
    for j in properties:
        clusterData = clusterData.loc[:, ~clusterData.columns.str.contains('^Unnamed')]
        DataFuel_j = clusterData[j] #count of fuel-j in cluster-i
        ax = sns.distplot(DataFuel_j, hist=True, kde=False, color = 'blue',hist_kws={'edgecolor':'black'},bins=10)
        try:
            os.mkdir('./IMGresult/param/'+str(folder)+'/'+str(i)+'/')
        except FileExistsError:
            pass
        plt.rc('text', usetex=True)
        fontsize=19
        plt.savefig('./IMGresult/param/'+str(folder)+'/'+str(i)+'/'+j+'.eps',dpi=600,orientation ='landscape')
        plt.close()

#converted individual plots
for i in fileName2:
    dataTaken = False
    ClusterWiseData = pd.DataFrame([])
    clusterData = pd.read_csv(path2+i+'.csv')
    count = 0
    for j in final_prop:
        clusterData = clusterData.loc[:, ~clusterData.columns.str.contains('^Unnamed')]
        DataFuel_j = clusterData[j] #count of fuel-j in cluster-i
        ax = sns.distplot(DataFuel_j, hist=True, kde=False, color = 'blue',hist_kws={'edgecolor':'black'},bins=10)
        try:
            os.mkdir('./IMGresult/param/'+str(folder)+'/'+str(i)+'/')
        except FileExistsError:
            pass
        plt.rc('text', usetex=True)
        fontsize=19
        plt.savefig('./IMGresult/param/'+str(folder)+'/'+str(i)+'/'+final_propName[count]+'.eps',dpi=600,orientation ='landscape')
        plt.close()
        count = count + 1


#Plot with only histogram
color = ['r','b','g','c','y','m','k','w']
for i in properties:
    PropertyWiseData = pd.DataFrame([])
    count = 0
    for j in fileName:
        clusterData = pd.read_csv(path+j+'.csv')
        clusterData = clusterData.loc[:, ~clusterData.columns.str.contains('^Unnamed')]
        DataFuel_j = clusterData[i] #count of fuel-j in cluster-i
        ax = sns.distplot(DataFuel_j, hist=True, kde=False, color = color[count],hist_kws={'edgecolor':'black'},bins=10,label=j)
        count = count +1
    plt.legend()
    try:
        os.mkdir('./IMGresult/ParaCombined/hist/')
    except FileExistsError:
        pass
    
    try:
        os.mkdir('./IMGresult/ParaCombined/hist/'+str(folder)+'/')
    except FileExistsError:
        pass
    plt.rc('text', usetex=True)
    fontsize=19
    plt.savefig('./IMGresult/ParaCombined/hist/'+str(folder)+'/'+str(i)+'.eps',dpi=600,orientation ='landscape')
    plt.close()

#Plot with histogram and pdf
color = ['r','b','g','c','y','m','k','w']
for i in properties:
    PropertyWiseData = pd.DataFrame([])
    count = 0
    for j in fileName:
        clusterData = pd.read_csv(path+j+'.csv')
        clusterData = clusterData.loc[:, ~clusterData.columns.str.contains('^Unnamed')]
        DataFuel_j = clusterData[i] #count of fuel-j in cluster-i
        ax = sns.distplot(DataFuel_j, hist=True, kde=True, color = color[count],hist_kws={'edgecolor':'black'},bins=10,label=j)
        count = count +1
    plt.legend()
    try:
        os.mkdir('./IMGresult/ParaCombined/hist_pdf/')
    except FileExistsError:
        pass
    
    try:
        os.mkdir('./IMGresult/ParaCombined/hist_pdf/'+str(folder)+'/')
    except FileExistsError:
        pass
    plt.rc('text', usetex=True)
    fontsize=19
    plt.savefig('./IMGresult/ParaCombined/hist_pdf/'+str(folder)+'/'+str(i)+'.eps',dpi=600,orientation ='landscape')
    plt.close()





