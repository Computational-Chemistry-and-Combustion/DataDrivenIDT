import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from common.search_fileNcreate import search_fileNcreate

def clusterParamaterAnalysis(curr_dir,dataset_location):
    path = curr_dir+'/result/cluster_data_before_regression/'
    files = os.listdir(path)
    fileName = [k.split('.')[0] for k in files]
    fName = ['Cluster-'+str(x) for x in range(len(files))]
    # data = pd.read_csv(dataset_location)
    # print(data.columns)
    properties = ['Equv(phi)','Fuel(%)','Oxidizer(%)','P(atm)','T(K)','Time(μs)']

    path2 = curr_dir+'/result/final_cluster/'
    # print(path2)
    files2 = os.listdir(path2)
    fileName2 = [k.split('.')[0] for k in files2]
    # print(fileName2)
    final_prop = ['log_P(atm)','log_Fuel(%)','log_Oxidizer(%)','T0/S_H__T','T0/T']
    final_propName = ['log_P(atm)','log_Fuel(%)','log_Oxidizer(%)','T0_S_H__T','T0_T']
    search_fileNcreate.check_directory(curr_dir+'/result/AnalysisResult/param/')


    #prop plots
    k=0
    for i in fileName:
        dataTaken = False
        ClusterWiseData = pd.DataFrame([])
        clusterData = pd.read_csv(path+i+'.csv')
        for j in properties:
            clusterData = clusterData.loc[:, ~clusterData.columns.str.contains('^Unnamed')]
            DataFuel_j = clusterData[j] #count of fuel-j in cluster-i
            ax = sns.distplot(DataFuel_j, hist=True, kde=False, color = 'blue',hist_kws={'edgecolor':'black'},bins=10)

            search_fileNcreate.check_directory(curr_dir+'/result/AnalysisResult/param/'+str(i)+'/')
            fontsize=15
            plt.xlabel(str(j),fontsize=fontsize)
            plt.ylabel("Count of data points",fontsize=fontsize)
            plt.title("Count of data points for "+ str(j) +" in "+fName[k],fontsize=fontsize)
            # plt.rc('text', usetex=True)
            # plt.legend( fName ,fontsize=fontsize-5,loc='best')
            plt.tight_layout()
            plt.savefig(curr_dir+'/result/AnalysisResult/param/'+str(i)+'/'+j+'.jpg',dpi=600,orientation ='landscape')
            plt.close()
        k = k+1

    # #converted individual plots
    k=0
    for i in fileName2:
        dataTaken = False
        ClusterWiseData = pd.DataFrame([])
        clusterData = pd.read_csv(path2+i+'.csv')
        count = 0
        for j in final_prop:
            clusterData = clusterData.loc[:, ~clusterData.columns.str.contains('^Unnamed')]
            DataFuel_j = clusterData[j] #count of fuel-j in cluster-i
            ax = sns.distplot(DataFuel_j, hist=True, kde=False, color = 'blue',hist_kws={'edgecolor':'black'},bins=10)
            
            search_fileNcreate.check_directory(curr_dir+'/result/AnalysisResult/param/'+str(i)+'/')
            fontsize=15
            plt.xlabel(str(j),fontsize=fontsize)
            plt.ylabel("Count of data points",fontsize=fontsize)
            plt.title("Count of data points for "+ str(j) +" in "+fName[k],fontsize=fontsize)
            # plt.rc('text', usetex=True)
            # plt.legend( fName ,fontsize=fontsize-5,loc='best')
            plt.tight_layout()
            plt.savefig(curr_dir+'/result/AnalysisResult/param/'+str(i)+'/'+final_propName[count]+'.jpg',dpi=600,orientation ='landscape')
            plt.close()
            count = count + 1
        k = k+1


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

        search_fileNcreate.check_directory(curr_dir+'/result/AnalysisResult/ParaCombined/hist/')
        fontsize=15
        plt.xlabel(str(i),fontsize=fontsize)
        plt.ylabel("Count of data points",fontsize=fontsize)
        plt.title("Count of data points for "+ str(i) +" in clusters",fontsize=fontsize)
        # plt.rc('text', usetex=True)
        # plt.legend( fName ,fontsize=fontsize-5,loc='best')
        plt.tight_layout()
        plt.savefig(curr_dir+'/result/AnalysisResult/ParaCombined/hist/'+str(i)+'.jpg',dpi=600,orientation ='landscape')
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
        
        search_fileNcreate.check_directory(curr_dir+'/result/AnalysisResult/ParaCombined/hist_pdf/')
        fontsize=15
        plt.xlabel(str(i),fontsize=fontsize)
        plt.ylabel("Density of data points",fontsize=fontsize)
        plt.title("Count of data points for "+ str(i) +" in clusters",fontsize=fontsize)
        # plt.rc('text', usetex=True)
        # plt.legend( fName ,fontsize=fontsize-5,loc='best')
        plt.tight_layout()
        plt.savefig(curr_dir+'/result/AnalysisResult/ParaCombined/hist_pdf/'+str(i)+'.jpg',dpi=600,orientation ='landscape')
        plt.close()





