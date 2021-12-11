#Calling libraries
import sys
import os 
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from common.reference_point import reference_point
from common.search_fileNcreate import search_fileNcreate as SF
from common.select_feature import select_feature as Sel_feat
from multiple_regression.regression import regression

#setting up paths
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)


#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
    Main_folder_dir += dir_split[i] + str('/')


def seperateClusterNregression(df,clusterIndex,tau,clusterNumber,originalData,curr_directory,limited_ref_points):
    '''
    This method will separate the cluster based on GMM assigned cluster 
    and perform regression over each cluster  
    and stores objects of ref points, cluster  and ...
    '''
    uniqueClusterID = np.unique(clusterIndex) #finding total unique cluster ID
    child_type = clusterNumber #sake of naming

    #finding and creating path
    SF.check_directory(str(curr_directory)+'/object_file/')
    SF.check_directory(str(curr_directory)+'/object_file/centroids/')

    #initialized object of ref points
    ref_point = reference_point(curr_directory,limited_ref_points=limited_ref_points)

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
        max_relerr_train, r2, testing_r2, summary,coefficients_dictionary, data = regression(dataset,y,curr_directory,cluster_label=i,test_size_fraction=0.05,elimination=False,child_type=i,sl=0.05,process_type ='tree') #find all the index-i having same uniqueClusterID[i]

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
        ref_point.other_reference_point(data,centroid,i,'cluster_ref')

        #final cluster gives stores only those cluster data which are useful fro prediction
        SF.check_directory(str(curr_directory)+'/result/final_cluster/')
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


# def Testing():
#     from data_gen import data_gen
#     external_data = pd.read_csv('testset.csv')
#     try: #if fuel data passed try this else skip and prcocess
#         list_fuel = find_fuel_type.find_strightchain_alkanes(external_data)
#         external_data = data_gen(external_data,list_fuel,Flag_value,curr_directory)     #normal dataset generation
#     except KeyError:
#         pass

#     #old
#     from old_external_test import old_external_test
#     testset_obj_old = old_external_test(Flag_value,curr_directory)
#     testset_obj_old.external_testset(external_data)

# def CycleTesting():
#     from data_gen import data_gen
#     external_data = pd.read_csv('FixedTest.csv')
#     try:
#         list_fuel = find_fuel_type.find_strightchain_alkanes(external_data)
#         dataset = data_gen(external_data,list_fuel,Flag_value,curr_directory)     #normal feature generation
#     except KeyError:
#         pass

#     #old
#     from old_external_test_cycle import old_external_test_cycle
#     testset_obj_old = old_external_test_cycle(Flag_value,curr_directory)
#     testset_obj_old.external_testset(dataset)

# def combineCluster():
#     from combined_N_analyze_all_test_result import combined_N_analyze_all_test_result
#     combined = combined_N_analyze_all_test_result(curr_directory)
#     combined.process()

# print("## GMM and data division for fuel data only## \n")



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