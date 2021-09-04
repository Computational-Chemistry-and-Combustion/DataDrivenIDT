#Calling libraries
import sys
import os 
import pandas as pd
import numpy as np
from find_fuel_type import find_fuel_type 
from reference_point import reference_point
from search_fileNcreate import search_fileNcreate as SF
import joblib
from select_feature import select_feature as Sel_feat
from data_gen import data_gen
# from Ternary_Tree import Ternary_Tree as TT
from sklearn.mixture import GaussianMixture 
from regression import regression
import matplotlib.pyplot as plt
#setting up paths
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)


#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
    Main_folder_dir += dir_split[i] + str('/')

#Inputs
Path = '/home/pragnesh/Git/Deploy_DDS/GMM/try/analysis'
Flag_value = '-t'
limited_ref_points = True
division_error_criteria = None

def seperateClusterNregression(df,clusterIndex,tau,clusterNumber,originalData):
    '''
    This method will seperate the cluuster based on GMM assigned cluster 
    and perform regression over each cluster  
    and stores objects of ref points, cluster  and ...
    '''
    uniqueClusterID = np.unique(clusterIndex) #finding total unique cluster ID
    child_type = clusterNumber #sake of naming

    #finding and creating path
    SF.check_directory(str(curr_directory)+'/object_file/')
    SF.check_directory(str(curr_directory)+'/object_file/centroids/')

    #initialized object of ref points
    ref_point = reference_point(curr_directory, division_error_criteria, Flag_value,limited_ref_points=limited_ref_points)

    #iterating over unique clusters
    for i in range(len(uniqueClusterID)):
        print('#########################################')
        print('##########Cluster: '+str(i)+'  ##################')
        print('#########################################')

        idx_list = np.where(clusterIndex == i) #list of index(s) equal to i
        # print(idx_list)
        clusterWiseOriginalData = originalData.iloc[idx_list]
        dataset = df.iloc[idx_list]
        y = tau.iloc[idx_list]

        SF.check_directory(str(curr_directory)+'/result/cluster_data_before_regression/')
        clusterWiseOriginalData.to_csv(str(curr_directory)+'/result/cluster_data_before_regression/cluster_'+str(i)+'.csv')

        #peforming regression
        max_relerr_train, r2, testing_r2, summary,coefficients_dictionary, data = regression(dataset,y,Flag_value,curr_directory,level = 0,cluster_label=i,test_size_fraction=0.05,elimination=False,child_type=i,sl=0.05,process_type ='tree') #find all the index-i having same uniqueClusterID[i]

        #save data of all clusters after regression 
        SF.check_directory(str(curr_directory)+'/result/cluster_data/')
        data.to_csv(str(curr_directory)+'/result/cluster_data/'+'/cluster_'+str(i)+'.csv')

        #calculating centroid
        centroid = ref_point.calculate_centroid(data) #pandas series
        SF.check_directory(str(curr_directory)+'/object_file/')
        SF.check_directory(str(curr_directory)+'/object_file/centroids/')

        ####object is stored
        filename_centroid=  str(curr_directory)+'/object_file/centroids/centroid_'+str(i)+'.sav'
        joblib.dump(centroid,filename_centroid)  
        
        # commented this part as it is required for final clusters and that we are going to
        # get after optimized cluster so no need to waste computation
        ref_point.other_reference_point(data,centroid,i,'tree_ref')

        #final cluster gives stores only those cluster data which are useful fro prediction
        SF.check_directory(str(curr_directory)+'/result/final_cluster/'+str(child_type))
        data.to_csv(str(curr_directory)+'/result/final_cluster/end_cluster_'+str(i)+'.csv')

        
        ##writing centroid
        ####For writing centroid
        SF.check_directory(str(curr_directory)+'/result/centroids/')
        centroid_headers = Sel_feat.column_selection().remove('Constant')
        try:
            centroid_out = pd.read_csv(str(curr_directory)+'/result/centroids/centroid_'+str(i)+'.csv')
        except pd.errors.EmptyDataError:
            centroid_out = pd.DataFrame([],columns=centroid_headers)
        except FileNotFoundError:
            centroid_out = pd.DataFrame([],columns=centroid_headers)
        centroid_out = centroid_out.append(pd.Series(centroid,index=centroid_headers),ignore_index=True)
        centroid_out.to_csv(str(curr_directory)+'/result/centroids/centroid_'+str(i)+'.csv',index=False)


def Testing():
    from data_gen import data_gen
    external_data = pd.read_csv('testset.csv')
    try: #if fuel data passed try this else skip and prcocess
        list_fuel = find_fuel_type.find_strightchain_alkanes(external_data)
        external_data = data_gen(external_data,list_fuel,Flag_value,curr_directory)     #normal dataset generation
    except KeyError:
        pass

    #old
    from old_external_test import old_external_test
    testset_obj_old = old_external_test(Flag_value,curr_directory)
    testset_obj_old.external_testset(external_data)


print("## GMM and data division for fuel data only## \n")


#finding out the straight chain alkanes
Fuel_data = pd.read_csv('trainset.csv')
list_fuel = find_fuel_type.find_strightchain_alkanes(Fuel_data)
dataset = data_gen(Fuel_data,list_fuel,Flag_value, Path)     #normal dataset generation
SF.check_directory(str(Path)+'/data/')
dataset.to_csv(str(Path)+'/data/full_data.csv')

#feature sepeation and selection
df,tau = Sel_feat.feature_selection(dataset)
     
till=3
aic_list =[]
bic_list = []
index = np.arange(1,till+1,1)
score_list = []
for clusterNumber in range(3,till+1):
    curr_directory = Path + '/Analysis_clusters_'+str(clusterNumber)
    gm = GaussianMixture(n_components=clusterNumber, random_state=0).fit(df,tau)
    clusterIndex = gm.predict(df)
    probability =  gm.predict_proba(df)
    pd.DataFrame(probability).to_csv('prob.csv')
    print('probability: ', probability)
    exit()
    aic_list.append(gm.aic(df))
    bic_list.append(gm.bic(df))
    score_list.append(gm.score(df))
    seperateClusterNregression(df,clusterIndex,tau,clusterNumber,Fuel_data)
    Testing()
    curr_directory = Path

print('\n\n Executed Normally! Please check plot Folder')
plt.rc('text', usetex=True)
fontsize=19
plt.plot(index,aic_list,'r-',label='AIC')
plt.plot(index,bic_list,'g-',label='BIC')
plt.title('GMM - AIC \& BIC criterion',fontsize=fontsize)
plt.xticks(index,fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('Number of clusters',fontsize=fontsize)
plt.ylabel('AIC/BIC',fontsize=fontsize)
plt.legend()
plt.tight_layout()
plt.savefig('AIC_BIC.eps',dpi=600,orientation ='landscape')
plt.show()

plt.plot(score_list,'b-',label='log-likelihood criterion')
plt.title('GMM - log-likelihood criterion',fontsize=fontsize)
plt.legend()
plt.tight_layout()
plt.savefig('log_likelihood.eps',dpi=600,orientation ='landscape')
plt.show()

### GMM demo
# import numpy as np
# from sklearn.mixture import GaussianMixture 
# import matplotlib.pyplot as plt
# # generate random data
# np.random.seed(1)
# n = 100
# x1 = np.random.uniform(0, 20, size=n)
# x2 = np.random.uniform(0, 20, size=n)
# x3 = np.random.uniform(0, 20, size=n)

# y1 = x1 + np.random.normal(size=n)
# y2 = 15 - x2 + np.random.normal(size=n)
# y3 = 10 + x3 + + np.random.normal(size=n)
# x = np.concatenate([x1, x2, x3])
# y = np.concatenate([y1, y2, y3])
# data = np.vstack([x, y]).T
# model = GaussianMixture(n_components=4).fit(data)
# plt.scatter(x, y, c=model.predict(data))
# plt.show()