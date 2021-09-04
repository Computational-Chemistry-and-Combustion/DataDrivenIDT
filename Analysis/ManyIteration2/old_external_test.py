######################### External test File  ##############################################
# This  file will call external testset and based using saved object classify the data 
# into clusters using centroid and do regression on classified data
#############################################################################################

import numpy as np
import pandas as pd 
import time 
import random
import joblib
from sklearn.model_selection import train_test_split
 ###Heat Map###
# import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import copy
# from data_gen import data_gen
import warnings
# from find_fuel_type import find_fuel_type
import subprocess
from search_fileNcreate import search_fileNcreate  as SF
##Directory to export the file of combination of different files
dir_path = './../'
from statsmodels.tools.eval_measures import rmse

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
# print('dir_path: ', dir_path)
sys.path.append(dir_path)
import math

#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
        Main_folder_dir += dir_split[i] + str('/')


class old_external_test():

        '''
        Checks new fuel which are not part of training or testing set
        '''
        def __init__(self,flag_value,curr_directory):
                self.flag_value = flag_value
                self.curr_directory = curr_directory
        


        def max_relative_error(self,y_train,y_train_pred):
                if(len(y_train) != 0): #if list is empty
                        error = np.max(np.abs(y_train - y_train_pred)/np.abs(y_train))
                        return error
                return 0 


        def external_testset(self,external_data):
                '''
                Adding test set from external source to predict new fuel 
                '''
                #finding out the straight chain alkanes
                warnings.warn('Processing only with straight chain Alkanes')
                try:
                        # '''
                        # If  externally features are supplied given more priorities
                        # '''
                        sys.path.append(self.curr_directory)
                        from feature_selection import select_feature as Sel_feat
                        print('feature selection passed')
                        
                except ImportError:
                        from select_feature import select_feature as Sel_feat
                        print('feature selection not passed')
                

                # print('In Select feature')
                df,tau = Sel_feat.feature_selection(external_data)
                # print('Out Select feature')
                
                # print('In finding total clusters')
                #finding number of clusters and file names of cluster object 
                directory_path = str(self.curr_directory)+'/result/cluster_data/'
                num_of_clusters, cluster_centroid_file_names, cluster_label = self.find_total_clusters(directory_path)
                
                print(num_of_clusters)
                print(cluster_centroid_file_names)
                print(cluster_label)

                # print('Out finding total clusters')

                #Classifying data based on clusters
                classified_df = self.assign_clusters_to_testdata(df,num_of_clusters,cluster_centroid_file_names,cluster_label)

                #Again joining the dependent value to the df, as it is easy to identify dependent variable when seperated by class
                df['Time(μs)'] = tau

                ###################################################################################
                ##################  For generating relative error in prediction ###################
                ###################################################################################
                error_result =[]
                final_comparision =pd.DataFrame([],columns=['y_actual','y_predicted','Relative Error'])
                
                testdata_points_in_cluster = []  #number of data points of testset in cluster
                rmse_cluster = [] #associated rmse of that cluster 

                ###linear regression
                for i in range(num_of_clusters):
                        #clustering process
                        #seperating data by class or cluster
                        cluster_dataframe = df[df['Class']  == cluster_label[i]] #cluster dataframe is obtained by single class
                        y_cluster = cluster_dataframe['Time(μs)']
                        cluster_dataframe_class = cluster_dataframe['Class']
                        cluster_dataframe = cluster_dataframe.drop(columns=['Time(μs)'])
                        cluster_dataframe = cluster_dataframe.drop(columns=['Class'])

                        #calling object files
                        regressor_OLS_modified = joblib.load(str(self.curr_directory)+'/object_file/tree/regressor/regressor_'+str(cluster_label[i])+'.sav')
                        X_names_modified = joblib.load(str(self.curr_directory)+'/object_file/tree/x_names/xname_'+str(cluster_label[i])+'.sav')
                        # scalar =  joblib.load(str(self.curr_directory)+'/object_file/scalar_'+str(cluster_label[i])+'.sav')
                        
                        #Dropping unnecessary features 
                        cluster_dataframe_header = list(cluster_dataframe.columns) 

                        # 1. first change drop the column which are not train set  by X_names
                        # 2. then transform the dataset #if any scalar is used


                        ###1.
                        for j in cluster_dataframe_header:
                                if(j not in X_names_modified ):
                                        cluster_dataframe = cluster_dataframe.drop(columns=j)
                                        # print('cluster_dataframe: ', cluster_dataframe.shape)
                        #transforming dataframe before adding one
                        #Generating list of headers with first column as constant of ones

                        # ###2. commented if no cluster is used
                        # cluster_dataframe = scalar.transform(cluster_dataframe)



                        #As Testing of all data points, all data points are assigned to the X_test_external
                        #testing to make compatible with dataframe 
                        X_test_external, y_test_external = cluster_dataframe.to_numpy() , y_cluster.to_numpy()
                        # print('X_test_external: ', X_test_external)
                        # print(regressor_OLS_modified.params)

                        y_pred = regressor_OLS_modified.predict(X_test_external)
                        cluster_dataframe['Class'] = cluster_dataframe_class
                        cluster_dataframe['Time(μs)_actual'] = y_test_external
                        cluster_dataframe['Time(μs)_predicted'] = y_pred
                        # result = loaded_model.score(X_test, Y_test)

                        SF.check_directory(str(self.curr_directory)+'/external_test_result/console_output/') #checking directory
                        SF.check_directory(str(self.curr_directory)+'/external_test_result/classified_data/') #checking directory
                        SF.check_file_existence(str(self.curr_directory)+"/external_test_result/console_output/output_result.txt")
                        cluster_dataframe.to_csv(str(self.curr_directory)+'/external_test_result/classified_data/classified_cluster'+str(cluster_label[i])+'.csv')
                        f = open(str(self.curr_directory)+"/external_test_result/console_output/output_result.txt", "a")
                        
                        ##########################################################
                        ###################  Result ##############################
                        ##########################################################

                        #resetting index
                        try:
                                y_pred = y_pred.reset_index(drop=True)
                                y_test_external = y_test_external.reset_index(drop=True)                                
                        except AttributeError:
                                pass
                        
                        ###printing result
                        print('\n\n\n\nResult for cluster-',str(cluster_label[i]),':\n')
                        f.write('\n Result for cluster-'+str(cluster_label[i])+':\n')
                        print('\n Index','          ','Y_actual','            ','Y_Predicted','                ','Relative Error')
                        f.write('\n Index'+'          '+'Y_actual'+'            '+'Y_Predicted'+'                '+'Relative Error')
                        for k,item in enumerate(y_pred):
                                # print(fuel_name[k],'            ',np.log(y_given[k]),'  ',y_test_external[k],'      ',y_pred[k],'    ',np.abs(y_test_external[k]-y_pred[k])/y_test_external[k],'\n')
                                print(k ,': ', y_test_external[k],'      ',y_pred[k],'    ',np.abs(y_test_external[k]-y_pred[k])/np.abs(y_test_external[k]),'\n')
                                f.write('\n'+str(k) +': '+ str(y_test_external[k])+'      '+str(y_pred[k])+'    '+str(np.abs(y_test_external[k]-y_pred[k])/np.abs(y_test_external[k]))+'\n')                        
                        ## result comparison and save
                        ID_comparison = pd.DataFrame()
                        ID_comparison['y_predicted'] = y_pred
                        ID_comparison['y_actual'] = y_test_external
                        ID_comparison['Relative Error'] = np.abs(y_pred - y_test_external)/np.abs(y_test_external)
                        # calc rmse
                        rmse_val = rmse(y_test_external, y_pred)
                        SF.check_directory(str(self.curr_directory)+'/external_test_result/Ignition_delay_comparison/') #checking directory
                        ID_comparison.to_csv(str(self.curr_directory)+'/external_test_result/Ignition_delay_comparison/ID_comparison_external_cluster_'+str(cluster_label[i])+'.csv')
                        maximum_relative_error_external = self.max_relative_error(y_test_external,y_pred)
                        f.write('\n\n Maximum Relative Error in external data for cluster-'+str(cluster_label[i])+' :'+str(maximum_relative_error_external))
                        print('\n\n Maximum Relative Error in external data for cluster-', str(cluster_label[i]),' :',str(maximum_relative_error_external))
                        print('\n\n Root Mean Square Error :', rmse_val)
                        f.write('\n\n Root Mean Square Error :'+str(rmse_val)+'\n\n')
                        f.close()
                        rmse_cluster.append(rmse_val)
                        testdata_points_in_cluster.append(len(y_pred))


                        ############## for overall relative error plot #################
                        final_comparision = pd.concat([final_comparision,ID_comparison],sort=True)

                        ###################
                        # ERROR PLOTTING  #
                        ###################

                        # '''
                        # plot of external test set result 
                        # '''
                        plt.close()
                        rel_error_gt_15 = ID_comparison[ID_comparison['Relative Error'] <= 0.15].shape[0]
                        rel_error_btn_15_20 = ID_comparison[(ID_comparison['Relative Error'] > 0.15) & (ID_comparison['Relative Error'] <= 0.20)].shape[0]
                        rel_error_btn_20_30 = ID_comparison[(ID_comparison['Relative Error'] > 0.20) & (ID_comparison['Relative Error'] <= 0.30)].shape[0]
                        rel_error_gt_30 = ID_comparison[ID_comparison['Relative Error'] > 0.30].shape[0]
                        x = ['$< 15\%$ ', '$ 15\% < x <= 20\%$','$ 20\% < x <= 30\%$','$ >30\%$']
                        y = [rel_error_gt_15,rel_error_btn_15_20,rel_error_btn_20_30,rel_error_gt_30]
                        SF.check_directory(str(self.curr_directory)+'/external_test_result/error_frequency/') #checking directory
                        plt.clf()
                        fontsize = 19
                        plt.bar(x,y)
                        plt.rc('text', usetex=True)
                        plt.grid(which='minor', alpha=0.2)
                        plt.title('Frequency of relative error in cluster -'+str(cluster_label[i]),fontsize=fontsize)
                        plt.xlabel('Relative Error',fontsize=fontsize)
                        plt.ylabel('Frequency of Error',fontsize=fontsize)
                        # plt.savefig(str(self.curr_directory)+'/external_test_result/error_frequency/error_frequency_'+str(cluster_label[i])+'.eps')
                        # plt.show()
                        plt.close()


                        #############
                        # PLOTTING  #
                        #############

                        #Drawing line at 45 
                        x = np.arange(-15,15,0.5)

                        # '''
                        # plot of external test set result 
                        # '''
                        SF.check_directory(str(self.curr_directory)+'/external_test_result/prediction_comparison_plots/') #checking directory
                        plt.clf()
                        plt.plot(x,x,linestyle='--',color='black')
                        plt.scatter(y_pred, y_test_external, s=20, cmap='viridis',label= str('Cluster-'+str(cluster_label[i]) +' data points'))
                        text = "Maximum Relative Error : "+ str(maximum_relative_error_external)
                        plt.xlim([2,11])
                        plt.ylim([2,11])
                        plt.rc('text', usetex=True)
                        plt.xlabel('Predicted IDT',fontsize=fontsize)
                        plt.ylabel('Actual IDT',fontsize=fontsize)
                        plt.tick_params(axis='both', which='major', labelsize=fontsize)
                        plt.text(3,0,text,)
                        plt.legend(loc='lower right',handlelength=1, borderpad=1.2, labelspacing=0.5,framealpha=0.5,fontsize=fontsize)
                        plt.tight_layout()
                        # plt.savefig(str(self.curr_directory)+'/external_test_result/prediction_comparison_plots/ignition_delay_external_'+str(cluster_label[i])+'.eps', format='eps', dpi=600)
                        plt.close()
        
                #Overall RMSE
                f = open(str(self.curr_directory)+"/external_test_result/console_output/output_result.txt", "a")
                # '''
                # overall rmse^2 * n = n1 * rmse1^2 + n2 * rmse2^2 +...s
                # '''
                f.write('rmse:'+str(rmse_cluster))
                f.write('data points in test cluster:'+str(testdata_points_in_cluster))
                square_rmse = 0
                for i,item in enumerate(rmse_cluster):
                        if(testdata_points_in_cluster[i] > 0): #if no data points then to avoid nan answer
                                square_rmse += (rmse_cluster[i]**2) * testdata_points_in_cluster[i]
                overall_rmse = math.sqrt(square_rmse / sum(testdata_points_in_cluster))
                f.write('\n \n Overall RMSE :'+str(overall_rmse))
                f.close()
                print('\n RMSE:',str(rmse_cluster))
                print('\n data points in test cluster:',str(testdata_points_in_cluster))
                print('\n \n Overall RMSE :',str(overall_rmse))

                #########################
                ### whole comparision ###
                #########################

                # '''
                # plot of external test set result 
                # '''
                rel_error_lt_10 = final_comparision[final_comparision['Relative Error'] <= 0.10].shape[0]
                rel_error_btn_10_20 = final_comparision[(final_comparision['Relative Error'] > 0.10) & (final_comparision['Relative Error'] <= 0.20)].shape[0]
                rel_error_btn_20_30 = final_comparision[(final_comparision['Relative Error'] > 0.20) & (final_comparision['Relative Error'] <= 0.30)].shape[0]
                rel_error_btn_30_40 = final_comparision[(final_comparision['Relative Error'] > 0.30) & (final_comparision['Relative Error'] <= 0.40)].shape[0]
                rel_error_btn_40_50 = final_comparision[(final_comparision['Relative Error'] > 0.40) & (final_comparision['Relative Error'] <= 0.50)].shape[0]
                rel_error_btn_50_60 = final_comparision[(final_comparision['Relative Error'] > 0.50) & (final_comparision['Relative Error'] <= 0.60)].shape[0]
                rel_error_btn_60_70 = final_comparision[(final_comparision['Relative Error'] > 0.60) & (final_comparision['Relative Error'] <= 0.70)].shape[0]
                rel_error_btn_70_80 = final_comparision[(final_comparision['Relative Error'] > 0.70) & (final_comparision['Relative Error'] <= 0.80)].shape[0]
                rel_error_btn_80_90 = final_comparision[(final_comparision['Relative Error'] > 0.80) & (final_comparision['Relative Error'] <= 0.90)].shape[0]
                rel_error_btn_90_100 = final_comparision[(final_comparision['Relative Error'] > 0.90) & (final_comparision['Relative Error'] <= 1.0)].shape[0]
                rel_error_gt_100 = final_comparision[(final_comparision['Relative Error'] > 1.0)].shape[0]

                # x = ['$<= 10\%$ ', '$ 10\% < x <= 20\%$','$ 20\% < x <= 30\%$','$ 30\% < x <= 40\%$','$ 40\% < x <= 50\%$','$ 50\% < x <= 60\%$','$ 60\% < x <= 70\%$','$ 70\% < x <= 80\%$','$ 80\% < x <= 90\%$','$ 90\% < x <= 100\%$','$ 100\% > x $']
                x = ['$10$', '$20$','$30$','$40$','$50$','$60$','$70$','$80$','$90$','$100 $',r'${{>}100}$']
                y = [rel_error_lt_10,rel_error_btn_10_20,rel_error_btn_20_30,rel_error_btn_30_40, rel_error_btn_40_50, rel_error_btn_50_60, rel_error_btn_60_70, rel_error_btn_70_80, rel_error_btn_80_90, rel_error_btn_90_100, rel_error_gt_100]
                SF.check_directory(str(self.curr_directory)+'/external_test_result/error_frequency/') #checking directory
                plt.clf()
                plt.bar(x,y)
                plt.rc('text', usetex=True)
                fontsize=19
                fontsize_ = 19
                plt.grid(which='minor', alpha=0.2)
                plt.title('Count of test-data points having \n relative error less than specified criteria',fontsize=fontsize)
                plt.xlabel('Relative Error ( $\le$  \%)',fontsize=fontsize)
                plt.ylabel('Count of test-data points',fontsize=fontsize)
                plt.xticks(fontsize=fontsize_, rotation=0)
                plt.yticks(fontsize=fontsize_, rotation=0)
                plt.tight_layout()
                plt.savefig(str(self.curr_directory)+'/external_test_result/error_frequency/error_frequency_all_data.eps', dpi=600)
                # plt.show()
                plt.close()                       
                # plt.show()
                plt.close()
                
                #Storing to the file 
                SF.check_file_existence(str(self.curr_directory)+'/external_test_result/error_frequency/error_frequency.csv')
                header = ['10','20','30','40','50','60','70','80','90','100','>100']
                try:
                        error_freq = pd.read_csv(str(self.curr_directory)+'/external_test_result/error_frequency/error_frequency.csv')
                except pd.errors.EmptyDataError:
                        error_freq = pd.DataFrame([],columns=header)
                new_data = pd.Series(y,index=header) #new error freq row
                error_freq = error_freq.append(new_data,ignore_index=True)    #appending
                error_freq.to_csv(str(self.curr_directory)+'/external_test_result/error_frequency/error_frequency.csv',index=False)


        def find_total_clusters(self,directory_path):
                '''
                This method will find out number of cluster based on center nodes saved on the 
                '''
                #counting number files in the centroid directory which is total number of centroids
                cmd_num_of_files = "find "+directory_path+" -type f | wc -l"
                #check_output return output of bash 
                num_of_cluster = int(subprocess.check_output(cmd_num_of_files,shell=True, universal_newlines=False))  # returns the exit code in unix

                #finding name of files in the centroid directory
                cmd_files_name = "ls "+directory_path
                centroid_file_names = str(subprocess.check_output(cmd_files_name,shell=True, universal_newlines=False),"utf-8").split('\n') #converting output into string and then splitting 
                file_names = [] #storing  file names
                file_name_label = []
                for i in range(len(centroid_file_names)-1):
                        file_names.append(centroid_file_names[i]) #storing file labels
                        # file_name_label.append(centroid_file_names[i][:-4].split('_')[-2]+'_'+centroid_file_names[i][:-4].split('_')[-1]) #storing centroid index -- cluster label #useful for other file reading
                        file_name_label.append(centroid_file_names[i][:-4].split('_')[-1]) #storing centroid index -- cluster label #useful for other file reading
                # file_names = file_names.sort()
                return num_of_cluster,file_names,file_name_label
        
        def assign_clusters_to_testdata(self,data,num_of_centroids,cluster_centroid_file_names,cluster_label):
                '''
                num_of_clusters = num_of_centroids
                calculating distance of data point from all the available centroid and appending the 
                calculated values from all clusters to new dataset - distance_from_centroids to find out the 
                least distance from the centroid and other reference points
                other reference points includes
                1. maximum distance from centroid
                2. minimum distance from centroid
                3. maximum distance from origin
                4. minimum distance from origin

                for given data distance will be measured for all the five point which denotes one cluster ]
                and procedure will be repeated for all the cluster 
                least distance will give final cluster to the point 
                
                for point-A,
                measure distance form all the five points of cluster-A SAVE ONLY LEAST DISTANCE
                measure distance form all the five points of cluster-B SAVE ONLY LEAST DISTANCE
                now based on least distance from A and B assign the cluster.
                '''
                data_passed = copy.deepcopy(data) #data processed for calculation assignment of centroid
                data_passed = data_passed.drop(columns=['Constant'])

                #finding distance from the centroid 

                classification_dataframe = pd.DataFrame([]) #converting into pandas DataFrame
                for i in range(num_of_centroids): #for all clusters
                        #reference points - contains all the data points from which distance has to be measured
                        ref_data_points = []
                        centroid = joblib.load(str(self.curr_directory)+'/object_file/centroids/centroid_'+cluster_label[i]+'.sav')
                        
                        ref_data_points.append(centroid)
                        # max_distance_from_centroid = joblib.load(str(self.curr_directory)+'/object_file/cluster_reference_points/maxCentroid_'+cluster_label[i]+'.sav')
                        # ref_data_points.append(max_distance_from_centroid)
                        # min_distance_from_centroid = joblib.load(str(self.curr_directory)+'/object_file/cluster_reference_points/minCentroid_'+cluster_label[i]+'.sav')
                        # ref_data_points.append(min_distance_from_centroid)
                        # max_distance_from_origin   = joblib.load(str(self.curr_directory)+'/object_file/cluster_reference_points/maxOrigin_'+cluster_label[i]+'.sav')
                        # ref_data_points.append(max_distance_from_origin)
                        # min_distance_from_origin   = joblib.load(str(self.curr_directory)+'/object_file/cluster_reference_points/minOrigin_'+cluster_label[i]+'.sav')
                        # ref_data_points.append(min_distance_from_origin)
                

                        m = 0
                        while(1):
                                try:
                                        dist_from_ref = joblib.load(str(self.curr_directory)+'/object_file/tree_ref/other_refPoi_'+str(m)+'_'+cluster_label[i]+'.sav' )  #cluster  label from i and  file sm 
                                        ref_data_points.append(dist_from_ref)
                                        m += 1
                                except FileNotFoundError:
                                        break

                        least_dist_from_all_reference = []
                        ########calculating euclidian distacne for all data points from each cluster
                        #distance of all data point from ref data points for one cluster
                        for j in range(len(data_passed)): #for all data points
                                distance_from_ref_points =[]
                                for k,item in enumerate(ref_data_points):
                                        distance_from_ref_points.append(self.euclidian_dist(data_passed.loc[j,:],ref_data_points[k]))#calling function
                                #minimum from above all
                                min_of_above_all = np.min(distance_from_ref_points)
                                least_dist_from_all_reference.append(min_of_above_all)
                        classification_dataframe[cluster_label[i]] = least_dist_from_all_reference
                        #extra
                        data[cluster_label[i]] = least_dist_from_all_reference
                #finding index of the minimum values and appending to the main dataframe
                data_class = classification_dataframe.idxmin(axis=1)
                # print('data_class: ', data_class)

                # Assigning centroid to the data 
                # note: data class is assigned based on centroid class
                data['Class'] = data_class #rather than asssigning number centroid name is assigned to the class
                return data

        def euclidian_dist(self,arr_1,arr_2):
                arr_1 = np.array(arr_1)
                arr_2 = np.array(arr_2)
                # '''
                # calculating distance by passed row of matrix and centroid 
                # '''
                distance = np.linalg.norm(arr_1-arr_2)
                return distance


if __name__ == "__main__": 
        external_data = pd.read_csv(str(Main_folder_dir)+'/data/test_set/test_dataset.csv')
        external_test.external_testset(external_data)
        # #manual result 
        # # cluster_result_format = ['intercept','Temp','pressure','fuel','oxygen','P_S','S_S','P_H','S_H']
        # cluster_0_coef ={'Constant':36.68,'Temp(K)':-15.99,'log_P(atm)':-0.55,'log_Fuel(%)':0.94, 'log_Oxidizer(%)':-1.69,'P_S':-0.28,'S_S':0.13,'P_H' :-0.84,'S_H' :0}
        # cluster_1_coef = {'Constant':32.96,'Temp(K)':-15.08,'log_P(atm)':-0.44,'log_Fuel(%)':0.68, 'log_Oxidizer(%)':-1.12,'P_S':-0.05,'S_S':0.29,'P_H' :-0.16,'S_H' :0}
        # cluster_2_coef ={'Constant':101.85,'Temp(K)':-42.46,'log_P(atm)':-1.04,'log_Fuel(%)':-0.14, 'log_Oxidizer(%)':-0.34,'P_S':-6.42,'S_S' :2.55,'P_H' :-19.26,'S_H': -1.31}
        # # cluster_2_coef ={'Constant':103.3123,'Temp(K)':-43.203,'log_P(atm)':-1.0753,'log_Fuel(%)':-0.225, 'log_Oxidizer(%)':-0.2724,'P_S':-6.4955,'S_S' :2.568,'P_H' :-19.4865,'S_H': -1.3595}

        # all_cluster_coef = ([cluster_0_coef,cluster_1_coef,cluster_2_coef])
        # external_test.manual_testing(external_data,all_cluster_coef)
