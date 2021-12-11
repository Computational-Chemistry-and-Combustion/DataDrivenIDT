import pandas as pd
import numpy as np
from multiple_regression.regression import regression
import warnings
import time
##Directory to export the file of combination of different files
dir_path = './../'
from tree.write_coef import writing_coefficient as WC
import sys
import os 
from common.search_fileNcreate import search_fileNcreate as SF
import copy
import joblib
import matplotlib.pyplot as plt
import matplotlib as rc
from common.reference_point import reference_point
dir_path = os.path.dirname(os.path.realpath(__file__))
# print('dir_path: ', dir_path)
sys.path.append(dir_path)


#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
        Main_folder_dir += dir_split[i] + str('/')

class Node():
    '''
    Defining Node along with its left and right child
    when object is created by this class then it will create 
    the left and right child automatically to store address
    It is created using help of two supportive queues.
    '''
    def __init__(self,data,y):
        '''
        constructor 
        '''
        self.data = data    # data
        self.y_dependent = y #dependent variable 

        # ''' Nodes are defined here where ever you see nodes defined changes accordingly in whole code'''
        self.left_node = None  #object left child
        self.right_node = None  #object right child
        self.center_node = None
        self.child_label = None
        self.r2 = None      #training R2 value 
        self.testing_r2 = None
        self.coefficients_dictionary = None
        self.data_size = None
        # self.uniq_fuel = None
        self.max_relerr_train = None
        self.centroid = None
    
class Ternary_Tree():

    def __init__(self,data_X,dependent_y,division_error_criteria,choice_value,curr_directory,cluster_label=0,elimination=False,sl=0.05,limited_ref_points=False):    #default testing criteria and do ctrl+f for more
        
        '''
        Constructor with root implementation 
        '''
                
        ##string to integer 
        self.division_error_criteria = float(division_error_criteria)
        self.choice_value = choice_value
        self.curr_directory = curr_directory
        self.cluster_label = cluster_label
        self.number_of_levels = 0
        self.elimination = elimination 
        self.sl = sl
        self.process_type = 'tree'

        #object of other module
        self.ref_point = reference_point(self.curr_directory,limited_ref_points=limited_ref_points)

        #root analysis
        self.root =Node(data_X,dependent_y) #root node defined 
        self.root.child_label = cluster_label
        #save data of all clusters after regression 
        SF.check_directory(str(self.curr_directory)+'/result/cluster_data_before_regression/'+str('root'))
        self.root.data.to_csv(str(self.curr_directory)+'/result/cluster_data_before_regression/'+str('root')+'/cluster_'+str(self.root.child_label)+'_'+str('root')+'.csv')
        
        self.root.max_relerr_train,self.root.r2,self.root.testing_r2,summary,self.root.coefficients_dictionary,self.root.data = regression(self.root.data,self.root.y_dependent,curr_directory=self.curr_directory,cluster_label=self.cluster_label,elimination=self.elimination,sl=self.sl,process_type=self.process_type)   #calculation of self.R2 of root 
        WC(self.root.coefficients_dictionary,self.root.r2,self.root.testing_r2,self.root.child_label,self.curr_directory,'root')
        self.root.data_size = self.root.data.shape[0]       #no of rows in data shows data points
        self.root.centroid = self.ref_point.calculate_centroid(self.root.data)
        # self.root.uniq_fuel = self.root.data['Fuel'].unique()     #for obtaing unique fuel in node
        self.root.coefficients_dictionary = self.remove_specific_char(self.root.coefficients_dictionary) #removing char for plotting #compatibility


    def Implement_Tree(self):

        if(self.root.data.empty is True):     #if root data sis none
            print('File is empty please check the file')    
        else:
            #NOTE : root data contains predicted values in the dataset  
            if(self.root.max_relerr_train > self.division_error_criteria):   #if r2 is less than division criteria divide the data
                self._divide(self.root.data,self.root) #passing the root as current node 
            else:
                # '''
                # Storing centroid of the center nodes as it gives final relation and useful for testing prediction
                # '''
                child_type = 'root'
                #checking directory 
                SF.check_directory(str(self.curr_directory)+'/object_file/')
                SF.check_directory(str(self.curr_directory)+'/object_file/centroids/')

                ####object is stored
                filename_centroid=  str(self.curr_directory)+'/object_file/centroids/centroid_'+str(self.root.child_label)+'.sav'
                joblib.dump(self.root.centroid,filename_centroid)  
                print('Criteria Satisfied')
                self.ref_point.other_reference_point(self.root.data,self.root.centroid,self.root.child_label)

                #final cluster gives stores only those cluster data which are useful fro prediction
                SF.check_directory(str(self.curr_directory)+'/result/final_cluster/'+str(child_type))
                self.root.data.to_csv(str(self.curr_directory)+'/result/final_cluster/'+str(child_type)+'/end_cluster_'+str(self.root.child_label)+'_'+str(child_type)+'.csv')

            # self.root.data.to_csv(str(self.curr_directory)+'/result/Data_of_Root.csv')
            
        print('Tree structure is implemented!')
        #calling method for printing
        self.LevelOrder()

 
    def _divide(self,data,cur_node):
        '''
        Recursive division
        Warning : if data points is less than 10 points in any child then it won't calculate 
        '''
        min_data_criterion = 0

        #Adding library 
        try:
                # '''
                # If  externally features are supplied given more prioritys
                # '''
                sys.path.append(self.curr_directory)
                from feature_selection import select_feature as Sel_feat
        except ImportError:
                from common.select_feature import select_feature as Sel_feat

        #dividing the data based on error and genearting 3 sets.

        data_left,y_left,data_center,y_center,data_right,y_right = self.division_by_error(data)

        # print('data_right: ', data_right.shape)
        # print('data_center: ', data_center.shape)
        # print('data_left: ', data_left.shape)
        # time.sleep(3)
        if(data_left.empty is False):   #if data is non-empty
            child_type = 'left'
            self.cluster_label += 1 
            if(cur_node.left_node is None):   #if current node of left child has no data 
                cur_node.left_node = Node(data_left,y_left)  #assign them dataset
                cur_node.left_node.data_size = cur_node.left_node.data.shape[0]
                cur_node.left_node.child_label = self.cluster_label
                print('\n\n  Cluster Node Label: ', self.cluster_label)
                # print('cur_node.left_node: ', cur_node.left_node.data.shape)
                # time.sleep(2)
                
                #save data of all clusters after regression 
                SF.check_directory(str(self.curr_directory)+'/result/cluster_data_before_regression/'+str(child_type))
                cur_node.left_node.data.to_csv(str(self.curr_directory)+'/result/cluster_data_before_regression/'+str(child_type)+'/cluster_'+str(cur_node.left_node.child_label)+'_'+str(child_type)+'.csv')
                
                if(cur_node.left_node.data.shape[0] > cur_node.left_node.data.shape[1]*2): #if rows are more than columns
                    cur_node.left_node.max_relerr_train, cur_node.left_node.r2, cur_node.left_node.testing_r2, summary,cur_node.left_node.coefficients_dictionary, cur_node.left_node.data = regression(cur_node.left_node.data,cur_node.left_node.y_dependent,curr_directory=self.curr_directory,cluster_label=cur_node.left_node.child_label,child_type=child_type,elimination=self.elimination,sl=self.sl,process_type=self.process_type)    #calculate the r2 of that data 
                    # print('max_relerr_train: ',cur_node.left_node.max_relerr_train)
                    WC(cur_node.left_node.coefficients_dictionary,cur_node.left_node.r2,cur_node.left_node.testing_r2,cur_node.left_node.child_label,self.curr_directory,child_type)
                    # time.sleep(1)
                    cur_node.left_node.data_size = cur_node.left_node.data.shape[0]             #for data size
                    # cur_node.left_node.uniq_fuel = cur_node.left_node.data['Fuel'].unique()   #for unique fuel in node                     cur_node.left_node.uniq_fuel = cur_node.left_node.data['Fuel'].unique()
                    cur_node.left_node.centroid = self.ref_point.calculate_centroid(cur_node.left_node.data)
                    #removing certain word from coefficients for printing
                    cur_node.left_node.coefficients_dictionary  = self.remove_specific_char(cur_node.left_node.coefficients_dictionary )
                    if(cur_node.left_node.max_relerr_train > self.division_error_criteria):    #if r2 is less than given criteria than divide further 
                            self._divide(cur_node.left_node.data,cur_node.left_node)
                    else:
                        # '''
                        # Storing centroid of the center nodes as it gives final relation and useful for testing prediction
                        # '''
                        #checking directory 
                        SF.check_directory(str(self.curr_directory)+'/object_file/')
                        SF.check_directory(str(self.curr_directory)+'/object_file/centroids/')
                        SF.check_directory(str(self.curr_directory)+'/object_file/leafs/')

                        ####object is stored
                        filename_centroid=  str(self.curr_directory)+'/object_file/centroids/centroid_'+str(cur_node.left_node.child_label)+'.sav'
                        joblib.dump(cur_node.left_node.centroid,filename_centroid)  
                        print('Criteria Satisfied')
                        
                        # commented this part as it is required for final clusters and that we are going to
                        # get after optimized cluster so no need to waste computation
                        self.ref_point.other_reference_point(cur_node.left_node.data,cur_node.left_node.centroid,cur_node.left_node.child_label,'cluster_ref')

                        #final cluster gives stores only those cluster data which are useful fro prediction
                        SF.check_directory(str(self.curr_directory)+'/result/final_cluster/'+str(child_type))
                        cur_node.left_node.data.to_csv(str(self.curr_directory)+'/result/final_cluster/'+str(child_type)+'/end_cluster_'+str(cur_node.left_node.child_label)+'_'+str(child_type)+'.csv')
                        print(cur_node.left_node.data)
                        cur_node.left_node.data.to_csv(str(self.curr_directory)+'/result/final_cluster/end_cluster_'+str(cur_node.left_node.child_label)+'.csv')

                        ####Storing the whole node
                        filename_centroid=  str(self.curr_directory)+'/object_file/leafs/leaf_'+str(cur_node.left_node.child_label)+'.sav'
                        joblib.dump(cur_node.left_node,filename_centroid)  

                        ##writing centroid
                        ####For writing centroid
                        SF.check_directory(str(self.curr_directory)+'/result/centroids/')
                        centroid_headers = Sel_feat.column_selection().remove('Constant')
                        try:
                            centroid_out = pd.read_csv(str(self.curr_directory)+'/result/centroids/centroid_'+str(cur_node.left_node.child_label)+'.csv')
                        except pd.errors.EmptyDataError:
                            centroid_out = pd.DataFrame([],columns=centroid_headers)
                        except FileNotFoundError:
                            centroid_out = pd.DataFrame([],columns=centroid_headers)
                        centroid_out = centroid_out.append(pd.Series(cur_node.left_node.centroid,index=centroid_headers),ignore_index=True)
                        centroid_out.to_csv(str(self.curr_directory)+'/result/centroids/centroid_'+str(cur_node.left_node.child_label)+'.csv',index=False)

                else:
                    print('Less Data')
            else:
                #if node has data then,
                if(self.cur_node.left_node.max_relerr_train > self.division_error_criteria):    #if r2 is less than given criteria than divide further 
                    if(cur_node.left_node.data.shape[0] > cur_node.left_node.data.shape[1]*2): #if rows are more than columns
                        self._divide(cur_node.left_node.data,cur_node.left_node)
                    else:
                        print('Error is more than given but doesn\'t contain enough data points in the cluster' )
                else:


                    print('Criteria Satisfied')
        
        # '''
        # If data is assigned to center then further rectification is not required.
        # Maximum relative Error will slightly increase from given criteria because of removal other points from center cluster
        # As removal of the points from reduces the numerator part (y_pred - y_act) reduced
        # '''
        if(data_center.empty is False):   #if data is non-empty
            child_type = 'center'
            self.cluster_label += 1 
            if(cur_node.center_node is None):   #if current node of left child has no data 
                cur_node.center_node = Node(data_center,y_center)  #assign them dataset
                cur_node.center_node.data_size = cur_node.center_node.data.shape[0]
                cur_node.center_node.child_label = self.cluster_label
                print('\n\n Cluster Node Label: ', self.cluster_label)
                # print('cur_node.center_node: ', cur_node.center_node.data.shape)
                # time.sleep(2)
                #save data of all clusters after regression 
                SF.check_directory(str(self.curr_directory)+'/result/cluster_data_before_regression/'+str(child_type))
                cur_node.center_node.data.to_csv(str(self.curr_directory)+'/result/cluster_data_before_regression/'+str(child_type)+'/cluster_'+str(cur_node.center_node.child_label)+'_'+str(child_type)+'.csv')

                if(cur_node.center_node.data.shape[0] > cur_node.center_node.data.shape[1]*2): #if rows are more than columns
                    cur_node.center_node.max_relerr_train, cur_node.center_node.r2, cur_node.center_node.testing_r2, summary,cur_node.center_node.coefficients_dictionary, cur_node.center_node.data = regression(cur_node.center_node.data,cur_node.center_node.y_dependent,curr_directory=self.curr_directory,cluster_label=cur_node.center_node.child_label,child_type=child_type,elimination=self.elimination,sl=self.sl,process_type=self.process_type)    #calculate the r2 of that data 
                    WC(cur_node.center_node.coefficients_dictionary,cur_node.center_node.r2,cur_node.center_node.testing_r2,cur_node.center_node.child_label,self.curr_directory,child_type)
                    # print('max_relerr_train: ',cur_node.center_node.max_relerr_train)
                    # time.sleep(1)
                    cur_node.center_node.data_size = cur_node.center_node.data.shape[0]             #for data size
                    # print('calculating centroid of center node ')
                    cur_node.center_node.centroid = self.ref_point.calculate_centroid(cur_node.center_node.data)
                    time.sleep(5)

                    # '''
                    # Storing centroid of the center nodes as it gives final relation and useful for testing prediction
                    # '''
                    #checking directory 
                    SF.check_directory(str(self.curr_directory)+'/object_file/')
                    SF.check_directory(str(self.curr_directory)+'/object_file/centroids/')
                    SF.check_directory(str(str(self.curr_directory)+'/object_file/leafs/'))

                    ####object is stored
                    filename_centroid=  str(self.curr_directory)+'/object_file/centroids/centroid_'+str(cur_node.center_node.child_label)+'.sav'
                    joblib.dump(cur_node.center_node.centroid,filename_centroid) 

                    ## commented this part as it is required for final clusters and that we are going to
                    ## get after optimized cluster so no need to waste computation
                    self.ref_point.other_reference_point(cur_node.center_node.data,cur_node.center_node.centroid,cur_node.center_node.child_label,'cluster_ref')

                  
                    #final cluster gives stores only those cluster data which are useful for prediction #as all centroid clusters are final clusters
                    SF.check_directory(str(self.curr_directory)+'/result/final_cluster/'+str(child_type)) 
                    cur_node.center_node.data.to_csv(str(self.curr_directory)+'/result/final_cluster/'+str(child_type)+'/end_cluster_'+str(cur_node.center_node.child_label)+'_'+str(child_type)+'.csv')
                    cur_node.center_node.data.to_csv(str(self.curr_directory)+'/result/final_cluster/end_cluster_'+str(cur_node.center_node.child_label)+'.csv')

                    ####Storing the whole node
                    filename_centroid=  str(self.curr_directory)+'/object_file/leafs/leaf_'+str(cur_node.center_node.child_label)+'.sav'
                    joblib.dump(cur_node.center_node,filename_centroid) 

                    ##writing centroid
                    ####For writing centroid
                    SF.check_directory(str(self.curr_directory)+'/result/centroids/')
                    centroid_headers = Sel_feat.column_selection().remove('Constant')
                    try:
                        centroid_out = pd.read_csv(str(self.curr_directory)+'/result/centroids/centroid_'+str(cur_node.center_node.child_label)+'.csv')
                    except pd.errors.EmptyDataError:
                        centroid_out = pd.DataFrame([],columns=centroid_headers)
                    except FileNotFoundError:
                            centroid_out = pd.DataFrame([],columns=centroid_headers)
                    centroid_out = centroid_out.append(pd.Series(cur_node.center_node.centroid,index=centroid_headers),ignore_index=True)
                    centroid_out.to_csv(str(self.curr_directory)+'/result/centroids/centroid_'+str(cur_node.center_node.child_label)+'.csv',index=False)


                    # '''
                    # Don't delete : useful in case if you want to further classify center nodes
                    # '''
            #         # cur_node.center_node.uniq_fuel = cur_node.center_node.data['Fuel'].unique()   #for unique fuel in node                     cur_node.center_node.uniq_fuel = cur_node.center_node.data['Fuel'].unique()
            
            #         #removing certain word from coefficients for printing
            #         cur_node.center_node.coefficients_dictionary  = self.remove_specific_char(cur_node.center_node.coefficients_dictionary )
                    if(cur_node.center_node.max_relerr_train > self.division_error_criteria):    #if r2 is less than given criteria than divide further 
                            # self._divide(cur_node.center_node.data,cur_node.center_node)
                            print('Error is slightly increased due to consideration of only center cluster data points')
                            
                    else:
                        print('Criteria Satisfied')
                else:
                    print('Less Data')
            # else:
            #     #if node has data then,
            #     if(self.cur_node.right_node.max_relerr_train > self.division_error_criteria):    #if r2 is less than given criteria than divide further 
            #         if(cur_node.right_node.data.shape[0] > cur_node.right_node.data.shape[1]*2): #if rows are more than columns
            #             # self._divide(cur_node.right_node.data,cur_node.center_node)
            #             pass
            #         else:
            #             print('Error is more than given but doesn\'t contain enough data points in the cluster' )
            #     else:
            #         print('Criteria Satisfied')

        
        if(data_right.empty is False):   #if data is non-empty
            child_type = 'right'
            self.cluster_label += 1 
            if(cur_node.right_node is None):   #if current node of left child has no data
                cur_node.right_node = Node(data_right,y_right)  #assign them dataset
                cur_node.right_node.data_size = cur_node.right_node.data.shape[0]
                cur_node.right_node.child_label = self.cluster_label
                print('\n\n Cluster Node Label: ', self.cluster_label)
                # print('cur_node.right_node: ', cur_node.right_node.data.shape)
                # time.sleep(2)
                #save data of all clusters after regression 
                SF.check_directory(str(self.curr_directory)+'/result/cluster_data_before_regression/'+str(child_type))
                cur_node.right_node.data.to_csv(str(self.curr_directory)+'/result/cluster_data_before_regression/'+str(child_type)+'/cluster_'+str(cur_node.right_node.child_label)+'_'+str(child_type)+'.csv')
                
                if(cur_node.right_node.data.shape[0] > cur_node.right_node.data.shape[1]*2): #if rows are more than columns
                    cur_node.right_node.max_relerr_train, cur_node.right_node.r2, cur_node.right_node.testing_r2, summary,cur_node.right_node.coefficients_dictionary, cur_node.right_node.data = regression(cur_node.right_node.data,cur_node.right_node.y_dependent,curr_directory=self.curr_directory,cluster_label=cur_node.right_node.child_label,child_type=child_type,elimination=self.elimination,sl=self.sl)    #calculate the r2 of that data 
                    WC(cur_node.right_node.coefficients_dictionary,cur_node.right_node.r2,cur_node.right_node.testing_r2,cur_node.right_node.child_label,self.curr_directory,child_type)
                    # print('max_relerr_train: ',cur_node.right_node.max_relerr_train)
                    # time.sleep(1)
                    cur_node.right_node.data_size = cur_node.right_node.data.shape[0]             #for data size
                    # cur_node.right_node.uniq_fuel = cur_node.right_node.data['Fuel'].unique()   #for unique fuel in node                     cur_node.right_node.uniq_fuel = cur_node.right_node.data['Fuel'].unique()
                    cur_node.right_node.centroid = self.ref_point.calculate_centroid(cur_node.right_node.data)
                    #removing certain word from coefficients for printing
                    cur_node.right_node.coefficients_dictionary  = self.remove_specific_char(cur_node.right_node.coefficients_dictionary )
                    
                    if(cur_node.right_node.max_relerr_train > self.division_error_criteria):    #if r2 is less than given criteria than divide further 
                            self._divide(cur_node.right_node.data,cur_node.right_node)
                    else:
                        # '''
                        # Storing centroid of the center nodes as it gives final relation and useful for testing prediction
                        # '''
                        #checking directory 
                        SF.check_directory(str(self.curr_directory)+'/object_file/')
                        SF.check_directory(str(self.curr_directory)+'/object_file/centroids/')
                        SF.check_directory(str(self.curr_directory)+'/object_file/leafs/')

                        ####object is stored
                        filename_centroid=  str(self.curr_directory)+'/object_file/centroids/centroid_'+str(cur_node.right_node.child_label)+'.sav'
                        joblib.dump(cur_node.right_node.centroid,filename_centroid) 
                        print('Criteria Satisfied')

                        ## commented this part as it is required for final clusters and that we are going to
                        ## get after optimized cluster so no need to waste computation  
                        self.ref_point.other_reference_point(cur_node.right_node.data,cur_node.right_node.centroid,cur_node.right_node.child_label,'cluster_ref')
                    
                        #final cluster gives stores only those cluster data which are useful fro prediction
                        SF.check_directory(str(self.curr_directory)+'/result/final_cluster/'+str(child_type)) 
                        cur_node.right_node.data.to_csv(str(self.curr_directory)+'/result/final_cluster/'+str(child_type)+'/end_cluster_'+str(cur_node.right_node.child_label)+'_'+str(child_type)+'.csv')
                        print(cur_node.right_node.data)
                        exit()
                        cur_node.right_node.data.to_csv(str(self.curr_directory)+'/result/final_cluster/end_cluster_'+str(cur_node.right_node.child_label)+'.csv',index=False)

                        ####Storing the whole node
                        filename_centroid=  str(self.curr_directory)+'/object_file/leafs/leaf_'+str(cur_node.right_node.child_label)+'.sav'
                        joblib.dump(cur_node.right_node,filename_centroid)  
                        
                        ##writing centroid
                        ####For writing centroid
                        SF.check_directory(str(self.curr_directory)+'/result/centroids/')
                        centroid_headers = Sel_feat.column_selection().remove('Constant')
                        try:
                            centroid_out = pd.read_csv(str(self.curr_directory)+'/result/centroids/centroid_'+str(cur_node.right_node.child_label)+'.csv')
                        except pd.errors.EmptyDataError:
                            centroid_out = pd.DataFrame([],columns=centroid_headers)
                        except FileNotFoundError:
                            centroid_out = pd.DataFrame([],columns=centroid_headers)
                        centroid_out = centroid_out.append(pd.Series(cur_node.right_node.centroid,index=centroid_headers),ignore_index=True)
                        centroid_out.to_csv(str(self.curr_directory)+'/result/centroids/centroid_'+str(cur_node.right_node.child_label)+'.csv',index=False)

                else:
                    print('Less Data')
            else:
                #if node has data then,
                if(self.cur_node.right_node.max_relerr_train > self.division_error_criteria):    #if r2 is less than given criteria than divide further 
                    if(cur_node.right_node.data.shape[0] > cur_node.right_node.data.shape[1]*2): #if rows are more than columns
                        self._divide(cur_node.right_node.data,cur_node.right_node)
                    else:
                        print('Error is more than given but doesn\'t contain enough data points in the cluster' )
                else:
                    print('Criteria Satisfied')
        

    def division_by_error(self,data):
        '''
        It will calculate the relative error of predicted and actual values of the all the data points.
        It will also generate the cluster(collect all the data point in one bin) of the which has relative 
        error less than specified criteria. 
        Apart from this, data points which has relative more than specified criteria will be collected in one bin
        and absolute error of such data points has been calculated.
        Data points which has positive sign for absolute error will be collected in one bin and data points which has
        negative sign will be collected in another bin. 
        so, overall three cluster of data points will be returned.(seperately dependent ans independent features)
        '''
        
        #finding the error 
        data['rel_error'] =  np.abs(data['y_act'] - data['y_pred'])/np.abs(data['y_act'])
        # print('data[]: ', data['rel_error'])

        #Clusters
        data_center = data[data['rel_error'] < self.division_error_criteria]
        data_outside_range = data[data['rel_error'] > self.division_error_criteria]
        self.points_in_range(data,self.division_error_criteria)

        #dropping unnecessary colm
        data_center = data_center.drop(columns=['rel_error'])
        data_outside_range = data_outside_range.drop(columns=['rel_error'])


        # print('cluster_1 size :',data_center.shape)
        # print('cluster_2 size :',data_outside_range.shape)

        data_outside_range['error_bifurcation'] =  data['y_act'] - data['y_pred']
        #left side data by positive error  - points on one side of the fit plane  beyond specified error 
        data_left = pd.DataFrame([])
        data_left =  data_outside_range.loc[data_outside_range['error_bifurcation'] >= 0] #+ve error

        #right side data by negative error  - points on another side of the fit plane beyond specified error 
        data_right = pd.DataFrame([])
        data_right =  data_outside_range.loc[data_outside_range['error_bifurcation'] < 0] #-ve error
        #dependent colms
        y_data = data['y_pred']
        y_left = data_left['y_act']
        y_right = data_right['y_act']
        y_center = data_center['y_act']
        self.cluster_plot(y_data,y_left,y_center,y_right)
        # time.sleep(10)
        #dropping extra columns after processing so child node have processable data 
        data_left = data_left.drop(columns=['error_bifurcation'])
        data_right = data_right.drop(columns=['error_bifurcation'])
        data_left = data_left.drop(columns=['y_act'])
        data_left = data_left.drop(columns=['y_pred'])
        data_right = data_right.drop(columns=['y_act'])
        data_right = data_right.drop(columns=['y_pred'])
        data_center = data_center.drop(columns=['y_act'])
        data_center = data_center.drop(columns=['y_pred'])



        return data_left,y_left,data_center,y_center,data_right,y_right
    
    def cluster_plot(self,data,data_left,data_center,data_right):
        '''
        This method will plot y_actual vs y_predcited values comparisons to show 
        left right and center clusters
        '''
        data_left_act  = data[data.index.isin(data_left.index)]
        # data_left_act.to_csv(str(self.curr_directory)+'/plots/cluster_plot_y/data_left_'+str(self.cluster_label)+'.csv')

        data_center_act = data[data.index.isin(data_center.index)]
        # data_center_act.to_csv(str(self.curr_directory)+'/plots/cluster_plot_y/data_center_'+str(self.cluster_label)+'.csv')

        data_right_act = data[data.index.isin(data_right.index)]
        # data_right_act.to_csv(str(self.curr_directory)+'/plots/cluster_plot_y/data_right_'+str(self.cluster_label)+'.csv')

        plt.clf()
        # use LaTeX fonts in the plot
        plt.rc('text', usetex=True)
        fontsize = 19
        # plt.rc('font', family='serif')
        plt.scatter(data_left_act,data_left,label='Left cluster',c='orange')
        plt.scatter(data_center_act,data_center,label='Center cluster',c='blue')
        plt.scatter(data_right_act,data_right,label='Right cluster',c='green')
        plt.xlabel('Actual value',fontsize=fontsize)
        plt.ylabel('Predicted value',fontsize=fontsize)
        plt.legend(fontsize=fontsize,loc = 'lower right')
        plt.xticks(np.arange(4,11,1),fontsize=fontsize, rotation=0)
        plt.yticks(np.arange(4,11,1),fontsize=fontsize, rotation=0)
        SF.check_directory(str(self.curr_directory)+'/plots/cluster_plot_y')
        plt.tight_layout()
        plt.savefig(str(self.curr_directory)+'/plots/cluster_plot_y/cluster_visulization_'+str(self.cluster_label)+'.eps')
        # plt.show()
        
    
   
    def points_in_range(self,data,defined_relative_error):
        '''
        shows how many data points lies within specified relative error 
        '''
        counter = 0
        total_data = data.shape[0]
        error_counter = np.sum(data['rel_error'] < defined_relative_error)

        print(error_counter,'out of ',total_data,',number of data point that has relative error less than ',defined_relative_error)

    def remove_specific_char(self,data_dictionary):
            '''
            TO MAKE COMPATIBLE WITH LATEX TIKZ
            to remove certain character like % and _ as they create problem while plotting tikz
            or one cad add \% adn \_ to print
            here i am removing.
            '''
            data_dictionary_values = data_dictionary.values()
            data_dictionary_keys = list(data_dictionary.keys())
            for k, item in enumerate(data_dictionary_keys):
                if(data_dictionary_keys[k] is not None):    
                    data_dictionary_keys[k] = data_dictionary_keys[k].replace('(%)','')
                    data_dictionary_keys[k] = data_dictionary_keys[k].replace('_','')
                    # print('dks: ', data_dictionary_keys[k])
            zipping = zip(data_dictionary_keys,data_dictionary_values)   #zipping the data 
            ###Defined diff dictionary so that they won't get manipulated by each other 
            modified_dictionary = dict(zipping)
            return modified_dictionary

    def R2(self,data):
        '''
        Checking R2 value 
        To test module and Check simple linear regression 
        '''
        ##Simple linear regression 
        temperature = data['T(K)']
        temperature = np.array(temperature).reshape(-1, 1)
        ignition_delay  = data['Time(μs)']

        from sklearn import linear_model
        from sklearn.metrics import r2_score
        reg = linear_model.LinearRegression().fit(temperature,ignition_delay)
        r2_simple = reg.score(temperature,ignition_delay)  
        # print('R2 value: ', r2_simple)

        return r2_simple

    def LevelOrder(self): #PrintTree
        '''
        This method prints all the data ay traversing level by level
        '''
        #defined current node as root
        curr_node = self.root

        # if (curr_node.data.empty() == True):
        #     print('Data is not properly implemented or Something is wrong with Dataset!')

        #Defining queue to store the data
        from collections import deque 
        ###Queues are defined to store children of node
        Q1 = deque() #queue-1 defined 
        Q2 = deque() #queue-2 defined 
         
        #Root Node data in queue 
        Q1.append(curr_node)
        # print('Root Data: ', curr_node.data)   #Shows root node
        self.number_of_levels = self.Calculate_levels(Q1,Q2) #calculating number fo levels in the tree
        self._print_LevelOrder(Q1,Q2) #for testing purpose only 

    def Calculate_levels(self,Q1,Q2):
        '''
        Level Order Printing by Traversing
        '''
        level = -1 #Tree level initial level zero

        # '''
        # Copy for generating the levels of the tree
        # '''
        
        print('''\n ##################### \
        \n # calculating levels# \
        \n ##################### ''')

        ###Queues are defined to store children of node
        copy_Q1 = copy.deepcopy(Q1) #queue-1 defined 
        copy_Q2 = copy.deepcopy(Q2) #queue-2 defined 
 
        # '''
        # For calculating levels of the tree
        # '''
        def append_if_not_None():
            pass
    

        while(len(copy_Q1) != 0 or len(copy_Q2) != 0):

            if(len(copy_Q1) != 0):
                #Level changed and queue also changed
                level += 1 #Tree level incremented
                # print('len(copy_Q2): ', len(copy_Q2))


            ##storing in the copy_Q2
            for i,item in enumerate(copy_Q1):
                if(copy_Q1[i] is not None):
                    if(copy_Q1[i].left_node is not None):
                        copy_Q2.append(copy_Q1[i].left_node)
                        # print('copy_Q1[i].left_node: ', copy_Q1[i].left_node)
                    if(copy_Q1[i].center_node is not None):
                        copy_Q2.append(copy_Q1[i].center_node)
                        # print('copy_Q1[i].center_node: ', copy_Q1[i].center_node)
                    if(copy_Q1[i].right_node is not None):
                        copy_Q2.append(copy_Q1[i].right_node)
                        # print('copy_Q1[i].right_node: ', copy_Q1[i].right_node)

            if( len(copy_Q2) != 0):
                #Level changed and queue also changed
                level += 1 #Tree level incremented
                # print('len(copy_Q2): ', len(copy_Q2))

            #Printing and Poping
            while(len(copy_Q1) != 0):
                copy_Q1.popleft()
                               
            ##storing in the copy_Q1
            for i,item in enumerate(copy_Q2):
                if(copy_Q2[i] is not None):
                    if(copy_Q2[i].left_node is not None):
                        copy_Q1.append(copy_Q2[i].left_node)
                        # print('copy_Q2[i].left_node: ', copy_Q2[i].left_node)
                    if(copy_Q2[i].center_node is not None):
                        copy_Q1.append(copy_Q2[i].center_node)
                        # print('copy_Q2[i].center_node: ', copy_Q2[i].center_node)
                    if(copy_Q2[i].right_node is not None):
                        copy_Q1.append(copy_Q2[i].right_node)
                        # print('copy_Q2[i].right_node: ', copy_Q2[i].right_node)

            #Printing and Poping
            while(len(copy_Q2) != 0):
                # i=0 #as after poping the 0th element 1st will become zero
                copy_Q2.popleft()
            


        #     '''{'family':'sans-serif','sans-serif':['Helvetica']})
        # ## for Palatino and other serif fonts use:
        # #rc('font',**{'family':'serif','serif':['Palatino']})
        # rc('text', usetex=True)
        #     just arrangement of numbers as final tree contains all none node and to remove such m\
        #     '''
        print('Total levels in the tree : ', level)
        level += 1  
        return level
         

   
    def _print_LevelOrder(self,Q1,Q2):

        '''
        Level Order Printing by Traversing
        As it is hard write the answer in the file but easy to handle in list as it has indexing :)
        '''
        #Change this value according to division of tree
        #works for any division
        ###############################
        type_of_division = 3

        # '''
        # Whereever you see three nodes defined change it to 
        # total defined nodes in node class
        # '''
        ###############################


        # Total_lines = 0     #total number of lines in the file 
        # #Calculating number of lines in the file 
        # for i in range(0,self.number_of_levels+1): #starting with 1 as 1 root node is directly added later
        #     Total_lines += (type_of_division^self.number_of_levels)
        
        tikz_array_training_r2 = [] #training data are stored here
        tikz_array_testing_r2 = [] #training data are stored here
        tikz_array_coefficient = [] #training data are stored here
        tikz_array_data_size= []    #trainign data sizes are stored here
        tikz_array_max_relerr_train = []      #for obtaining unique fuels in node
        tikz_array_child_index =[]
        #counter to keep track of index of data 
        counter = 1
        counter_list =[] # counter list to appended the data

        #Initializing dictionary with none
        #generating list and finally dictionary 
        for i in range(self.number_of_levels+1): 
            list_to_append = [None]*(type_of_division**i) #list to generate the dictionary
            # print('list_to_append: ', list_to_append)
            ####loop to generate the number so using this number we can easily controll the data
            for j in range(len(list_to_append)):
                counter_list.append(counter)            
                counter += 1
            # print('counter_list: ', counter_list)
            zipping = zip(counter_list,list_to_append)    #zipping the data ###Initialized with None values
            ###Defined diff dictionary so that they won't get manipulated by each other 

            dictionary_None = dict(zipping) #making dictionary of None and 
            counter_list =[]
            tikz_array_training_r2.append(copy.deepcopy(dictionary_None)) #tikz_array_training_r2 is final resulting array 
            tikz_array_testing_r2.append(copy.deepcopy(dictionary_None)) #training data are stored here
            tikz_array_coefficient.append(copy.deepcopy(dictionary_None)) #training data are stored here
            tikz_array_data_size.append(copy.deepcopy(dictionary_None))  #training  data size are store here
            tikz_array_max_relerr_train.append(copy.deepcopy(dictionary_None))  #uniq fuels in training data 
            tikz_array_child_index.append(copy.deepcopy(dictionary_None))

        # '''
        # Have more than one data to print can I zip other all data into dictionary 
        # ''' 
        Tree_data_to_print = [] #Tree data to zip in the one part and store as list 

        #to keep track of counter
        dictionary_index_counter = 1

        def print_shape(node):
            if(node is not None):
                print('Node data : ',node.data.shape)


        #initializing level as before 
        level = -1 #Tree level initial level zero
        while(len(Q1) != 0 or len(Q2) != 0):
            
            level += 1 #Tree level incremented

            #if level increment goes ahead then defined has to break 
            #for safety measure as none is added every time 
            if(level > self.number_of_levels):
                    break

            ##storing in the Q2
            for i,item in enumerate(Q1):
                if(Q1[i] is not None):
                    # print(str(Q1[i].r2))
                    Q2.append(Q1[i].left_node)
                    # print('Q1[i].left_node: ', Q1[i].left_node)
                    # print_shape(Q1[i].left_node)
                    Q2.append(Q1[i].center_node)
                    # print('Q1[i].center_node: ', Q1[i].center_node)
                    # print_shape(Q1[i].center_node)
                    Q2.append(Q1[i].right_node)
                    # print('Q1[i].right_node: ', Q1[i].right_node)
                    # print_shape(Q1[i].right_node)
                else:
                    Q2.append(None)
                    Q2.append(None)             
                    Q2.append(None)

                
            #Printing and Poping
            while(len(Q1) != 0):
                i=0
                if(Q1[i] is not None):
                    tikz_array_training_r2[level][dictionary_index_counter] = Q1[i].r2
                    tikz_array_testing_r2[level][dictionary_index_counter] = Q1[i].testing_r2
                    tikz_array_coefficient[level][dictionary_index_counter] = Q1[i].coefficients_dictionary
                    tikz_array_data_size[level][dictionary_index_counter] = Q1[i].data_size
                    tikz_array_max_relerr_train[level][dictionary_index_counter] = Q1[i].max_relerr_train
                    tikz_array_child_index[level][dictionary_index_counter] = Q1[i].child_label
                else:
                    tikz_array_training_r2[level][dictionary_index_counter] = None
                    tikz_array_testing_r2[level][dictionary_index_counter] = None
                    tikz_array_coefficient[level][dictionary_index_counter] = None
                    tikz_array_data_size[level][dictionary_index_counter] = None
                    tikz_array_max_relerr_train[level][dictionary_index_counter] = None
                    tikz_array_child_index[level][dictionary_index_counter] = None

                Q1.popleft()
                dictionary_index_counter += 1



            ###############################################
            ###############################################
            #For Q2
            #Level changed and queue also changed
            # print("\n")
            level += 1 #Tree level incremented
            # print('level: ', level)
            # print('\n')
            
            #if level increment goes ahead then defined has to break 
            #for safety measure as none is added every time
            if(level > self.number_of_levels):
                break


            ##storing in the Q1
            counter = 0
            for i,item in enumerate(Q2):
                if(Q2[i] is not None):
                    Q1.append(Q2[i].left_node)
                    # print('Q2[i].left_node: ', Q2[i].left_node)
                    # print_shape(Q2[i].left_node)
                    Q1.append(Q2[i].center_node)
                    # print('Q1[i].center_node: ', Q2[i].center_node)
                    # print_shape(Q2[i].center_node)
                    Q1.append(Q2[i].right_node)
                    # print('Q2[i].right_node: ', Q2[i].right_node)
                    # print_shape(Q2[i].right_node)

                else:
                    Q1.append(None)
                    Q1.append(None)
                    Q1.append(None)
                                
                
            #Printing and Poping
            while(len(Q2) != 0):
                i=0 #as after poping the 0th element 1st will become zero
                if(Q2[i] is not None):
                    tikz_array_training_r2[level][dictionary_index_counter] = Q2[i].r2
                    tikz_array_testing_r2[level][dictionary_index_counter] = Q2[i].testing_r2
                    tikz_array_coefficient[level][dictionary_index_counter] = Q2[i].coefficients_dictionary
                    tikz_array_data_size[level][dictionary_index_counter] = Q2[i].data_size
                    tikz_array_max_relerr_train[level][dictionary_index_counter] = Q2[i].max_relerr_train
                    tikz_array_child_index[level][dictionary_index_counter] = Q2[i].child_label
                else:
                    tikz_array_training_r2[level][dictionary_index_counter] = None
                    tikz_array_testing_r2[level][dictionary_index_counter] = None
                    tikz_array_coefficient[level][dictionary_index_counter] = None
                    tikz_array_data_size[level][dictionary_index_counter] = None
                    tikz_array_max_relerr_train[level][dictionary_index_counter] = None
                    tikz_array_child_index[level][dictionary_index_counter] = None       
                    
                Q2.popleft()
                dictionary_index_counter += 1

            #if level increment goes ahead then defined has to break 
            #for safety measure as none is added every time
            if(level > self.number_of_levels):
                break

        # print('\n \n  \n \n \n ')
        # print('tikz_array_training_r2: ', tikz_array_training_r2)
        # print('tikz_array_testing_r2: ', tikz_array_testing_r2)
        # print('tikz_array_coefficient: ', tikz_array_coefficient)
        # print('tikz_array_data_size: ', tikz_array_data_size)
        # ###Processing and Printing the training data

        latex_supported_array = self.processing_list_to_tikz_file(tikz_array_child_index,type_of_division)        
        # print('latex_supported_array: ', latex_supported_array)
        self.write_latex(latex_supported_array,type_of_division,filename = 'ChildLabel',)

        latex_supported_array = self.processing_list_to_tikz_file(tikz_array_training_r2,type_of_division)        
        # print('latex_supported_array: ', latex_supported_array)
        self.write_latex(latex_supported_array,type_of_division,filename = 'Training',)
        
        ###Processing and Printing the testing data
        latex_supported_array = self.processing_list_to_tikz_file(tikz_array_testing_r2,type_of_division)        
        # print('latex_supported_array: ', latex_supported_array)
        self.write_latex(latex_supported_array,type_of_division,filename = 'Testing')
         
        ###Processing and Printing the coefficient data
        latex_supported_array = self.processing_list_to_tikz_file(tikz_array_coefficient,type_of_division)        
        # print('latex_supported_array: ', latex_supported_array)
        self.write_latex(latex_supported_array,type_of_division,filename = 'coefficient')

        ###Processing and Printing the Datasize
        latex_supported_array = self.processing_list_to_tikz_file(tikz_array_data_size,type_of_division)        
        # print('\nlatex_supported_array: ', latex_supported_array)
        self.write_latex(latex_supported_array,type_of_division,filename = 'Datasize')

        ###Processing and Printing the FuelsTrainingTesting
        latex_supported_array = self.processing_list_to_tikz_file(tikz_array_max_relerr_train,type_of_division)        
        # print('latex_supported_array: ', latex_supported_array)
        self.write_latex(latex_supported_array,type_of_division,filename = 'MaxRelError')
        
    
    def processing_list_to_tikz_file(self,tikz_array,type_of_division):
        '''
        converting array in format that is compatible for printing
                        1
                2               3
            4       5       6       7   
         8     9 10   11 12   13 14   15    

         which can be written as which also useful for latex tikZ,
         1
            2
                4
                    8
                    9
                5
                    10
                    11
            3
                6
                    12
                    13
                7
                    14  
                    15
        by level oredering traversing ,
        result will be : 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 
        but to print in tikz(latex) in should be ,
        result should be : 1 2 4 8 9 5 10 11 3 6 12 13 7 14 15
        by following pattern(9/(division_brach=2) = 4.5 < 5 ) it is implemented.
        input : level order traversing array 
        output : tikz compatible error
        '''
        # print('processing_list_to_tikz_file: ')
        # tikz_data_list_modified = tikz_data_list.copy()
        # tikz_data_list_modified.insert(0,'None')
        

        #A None array inserted to sake of easiness DIVISION^i
        tikz_array_copy = tikz_array.copy()###MAKE A COPY AND PROCEED

        tikz_array_modified = dict()   #it's dictionary

        total_elements  =0
        for i in range(self.number_of_levels+1):
            total_elements = type_of_division ** i

        #unrolling many dictionary into one dictionary 
        # EX:
        # [{1:a},{2:b,3:c},{4:d,5:e,6:f,7:g}]
        # {1:a,2:b,3:c,4:d,5:e,6:f,7:g}
        element_counter = 1 #it's also keys for dictionary 
        for i in range(self.number_of_levels+1): #to iterate over levels
            elements  = type_of_division ** i
            for j in range(elements):
                key = element_counter
                value  = tikz_array[i].get(element_counter) #ith level and jth dict element 
                tikz_array_modified.update({key:value})
                element_counter += 1 

        #taking initial values in the array 
        #EX: 
        # for binary tree  : [1, 2, 4, 8, 16, 32, 64, 128, 256]
        #for tertiary tree : [1, 2,  5, 14, 41, 122, 365, 1094, 3281]
        #Note :  check for type_of_division = 2,3 it should work for any tree 
        key_index_array = [] #first keys are extreme left values of tree like 1 2 4 8
        key_index_array.append(0) #first index appended

        for i in range(1,self.number_of_levels): #+1 can include last level
            #3^0=1  -> (previous)0+3^0=1    -> next index 1  #first index assign
            #3^1=3  -> (previous)1+3^1=4    -> +1 next index 5 
            #3^2=9  -> (previous)4+3^2=13   -> +1 next index 14
            #3^3=27 -> (previous)13+3^3=40  -> +1 next index 41
            #formula :  previous_key + 3^i + 1
            key_index = (key_index_array[i-1]) + (type_of_division**i) 
            key_index_array.append(key_index)
        key_index_array = [x+2 for x in key_index_array]
        key_index_array[0] = 1 #first index is 1 so
        key_index_array.insert(1,2) #index 2 appending 

        #using above initial label updating the 
        def control_function(num,type_of_division):
            '''
            This method calculates control formula which is useful for traversing in the tree
            '''
            return num*type_of_division+1


        def check_forward(control_value,control_index ,key_index_array,tikz_array_modified,tikz_array_required,upper_bound):
            '''
            A.
            THis method will append value till second last element
            starting with second last element--
            array : [1,2,5,14,41]
            map : [4,7,16,43,XX]  formula : 3^i+1

            increment till array values 41 -> 42 -> 43 -> 44

            as 44 of array  > 43 of map so, traverse back 

            control on 3rd element 
            1. if second last element print all child
            2. after printing go back till control value is greater function changed 
            go to B.

            C. traverse forward increment every element by one and also append in final list. Repeat till second last element.
            array : [1,2,5,15,44]
            map : [4,7,16,46,XX]

            as now second last element print all child and go to step A.

            '''
            # print('control_value: ', control_value)
            while(control_index < len(key_index_array)-2):
                key_index_array[control_index+1] += 1
                # print('key_index_array: ', key_index_array)
                key = key_index_array[control_index+1]
                # print('key: ', key)
                value = tikz_array_modified[key]
                # print('value: ', value)
                tikz_array_required.update({key:value})
                # print('tikz_array_required: ', tikz_array_required)
                control_index += 1
            # print('key_index_array: ', key_index_array)
            # print('tikz_array_required: ', tikz_array_required)
            if(control_index == len(key_index_array)-2): #starts with 0 and 1 less so -2
                for i in range(type_of_division):
                    key = key_index_array[control_index+1]
                    # print('key: ', key)
                    value = tikz_array_modified[key]
                    # print('value: ', value)
                    tikz_array_required.update({key:value})
                    # print('tikz_array_required: ', tikz_array_required)
                    key_index_array[control_index+1] += 1
                    # print('key_index_array: ', key_index_array)
                #if upper bound reached return 
                if(key == upper_bound):
                    return
            # print('tikz_array_required: ', tikz_array_required)
            check_backward(control_value,control_index,key_index_array,tikz_array_modified,tikz_array_required,upper_bound)

        def check_backward(control_value,control_index,key_index_array,tikz_array_modified,tikz_array_required,upper_bound):
            '''
            B.
            for tertiary,
            array : [1,2,5,14,44]
            map : [4,7,16,43,XX]

            43 < 44
            but 
            16 !< 14
            so, control on second 
            go to C.


            '''
            while(control_value <= key_index_array[control_index+1]):
                control_index -= 1
                # print('control_index: ', control_index)
                control_value = control_function(key_index_array[control_index],type_of_division)
                # print('control_value: ', control_value)
            # print('control_index: ', control_index)
            # print('control_value: ', control_value)
            # print('key_index_array: ', key_index_array)
            check_forward(control_value,control_index,key_index_array,tikz_array_modified,tikz_array_required,upper_bound)
            #if upper bound reached return 
            if(key == upper_bound):
                return
        
        #final output empty dict
        tikz_array_required = dict()
        control_index = self.number_of_levels-1

        upper_bound = 0
        for i in range(self.number_of_levels+1):
            upper_bound += type_of_division**i

        for i in range(self.number_of_levels):
            key = key_index_array[i]
            value = tikz_array_modified.get(key)
            tikz_array_required.update({key:value})

        while(key_index_array[control_index+1] < upper_bound):
            control_value = control_function(key_index_array[control_index],type_of_division) 
            check_forward(control_value,control_index ,key_index_array,tikz_array_modified,tikz_array_required,upper_bound)

        return tikz_array_required

    def write_latex(self,latex_supported_array,type_of_division,filename = 'tree'):
        '''
        Note : if you change tree type_of_division please change closing brackets accordingly
        Note: also set the distance based on latex 
        Writing latex file with help of 
        '''
        ##############################
        # Writing texplot to the file#
        ##############################
        # print('last   :'  ,latex_supported_array)
        SF.check_directory(str(self.curr_directory)+"/plots/")
        f = open(str(self.curr_directory)+"/plots/"+str(filename)+".tex", "w")
        # A simple Tree
        # Author: Pragnesh Rana
        f.write(" \documentclass[a4paper,10pt]{article}\n")
        f.write(" \\usepackage{tikz}\n")
        f.write(" \\usepackage{fullpage}\n")
        f.write(" \\usetikzlibrary{positioning,shadows,arrows,trees,shapes,fit}\n")
        f.write(" \\begin{document}\n")
        f.write(" \\begin{figure}\n")
        f.write(" \\begin{tikzpicture}\n")
        f.write(" [font=\small, edge from parent fork down, \n")
        f.write(" every node/.style={top color=white, bottom color=blue!25, \n")
        f.write(" 	rectangle,rounded corners, minimum size=5mm, draw=blue!75,\n")
        f.write("	very thick, drop shadow, align=center},\n")
        f.write(" edge from parent/.style={draw=blue!50,thick},\n")
        f.write(" level 1/.style={sibling distance=4cm},\n")
        if(filename == 'coefficient'):
            f.write(" level 2/.style={sibling distance=4cm}, \n")
            f.write(" level 3/.style={sibling distance=4cm}, \n")
            f.write(" level 4/.style={sibling distance=4cm}, \n")
            f.write(" level 5/.style={sibling distance=4cm}, \n")
            f.write(" level 6/.style={sibling distance=4cm}, \n")
            f.write(" level distance=2cm,\n")
        else:
            f.write(" level 2/.style={sibling distance=3cm}, \n")
            f.write(" level 3/.style={sibling distance=3cm}, \n")
            f.write(" level 4/.style={sibling distance=2.5cm}, \n")
            f.write(" level 5/.style={sibling distance=2.5cm}, \n")
            f.write(" level 6/.style={sibling distance=2cm}, \n")
            f.write(" level 7/.style={sibling distance=2cm}, \n")
            f.write(" level 8/.style={sibling distance=2cm}, \n")
        if(filename == 'FuelsTrainingTesting' or filename == 'coefficient'):
            f.write(" level distance=5cm,\n")
        else:
            f.write(" level distance=2cm,\n")
        f.write(" ]\n")

        # if(filename == 'coefficient'):
        #     for i in range(len(latex_supported_array)):

        #coding part to write array 
        len_array = len(latex_supported_array)
        # print('len_array: ', len_array)
        key_list_latex_array = list(latex_supported_array.keys())
        # print('key_list_latex_array: ', key_list_latex_array)
        output_value_list = list(latex_supported_array.values())
        # print('output_value_list: ', output_value_list)
        # '''
        # Don't delete the commented code 
        # # '''
        # ##To print just a tree structure  with numbers (uncomment) with "None" values
        # f.write('\\node {'+str(key_list_latex_array[0])+'} %root\n')
        # for i in range(1,len_array):
        #     f.write('child { node {'+ str(key_list_latex_array[i])+'}') 
        #     if( i < len_array-2):
        #         '''
        #         if two keys are greater than current then do nothing 
        #         if first is greater and second is less than one bracket 
        #         if both less then two brackets
        #         '''
        #         if(key_list_latex_array[i+1] > key_list_latex_array[i] and key_list_latex_array[i+2]  > key_list_latex_array[i]):
        #             f.write(' \n')
        #             continue
        #         elif(key_list_latex_array[i+1] > key_list_latex_array[i] and key_list_latex_array[i+2]  < key_list_latex_array[i]):
        #             f.write(' }\n')
        #             continue
        #         elif(key_list_latex_array[i+1] < key_list_latex_array[i] ):#and key_list_latex_array[i+2]  < key_list_latex_array[i]):
        #             f.write(' }\n')
        #             j=i; #index defined for another loop
        #             while(key_list_latex_array[j+1] < key_list_latex_array[i]):
        #                 f.write(' }\n')
        #                 j +=  1
        #             continue
        #     elif(i == len_array-2):
        #         f.write('}\n')
        #         continue
        #     elif(i == len_array-1):
        #         f.write('}\n')
        #         for i in range(self.number_of_levels-2):
        #             f.write('}\n')
        #         continue


        # '''
        # To r2 value of training data with "None" values
        # '''
        # ##To print just a tree structure with R2 value with "None" values uncomment
        # f.write('\\node {'+str(output_value_list[0])+'} %root\n')
        # for i in range(1,len_array):
        #     f.write('child { node {'+ str(output_value_list[i])+'}') 
        #     if( i < len_array-2):
        #         '''
        #         if two keys are greater than current then do nothing 
        #         if first is greater and second is less than one bracket 
                # if both less then two brackets
        #         '''
        #         if(key_list_latex_array[i+1] > key_list_latex_array[i] and key_list_latex_array[i+2]  > key_list_latex_array[i]):
        #             f.write(' \n')
        #             continue
        #         elif(key_list_latex_array[i+1] > key_list_latex_array[i] and key_list_latex_array[i+2]  < key_list_latex_array[i]):
        #             f.write(' }\n')
        #             continue
        #         elif(key_list_latex_array[i+1] < key_list_latex_array[i] ):#and key_list_latex_array[i+2]  < key_list_latex_array[i]):
        #             f.write(' }\n')
        #             j=i; #index defined for another loop
        #             while(key_list_latex_array[j+1] < key_list_latex_array[i]):
        #                 f.write(' }\n')
        #                 j +=  1
        #             continue
        #     elif(i == len_array-2):
        #         f.write('}\n')
        #         continue
        #     elif(i == len_array-1):
        #         f.write('}\n')
        #         for i in range(self.number_of_levels-2):
        #             f.write('}\n')
        #         continue


        # '''
        # To r2 value of training data
        # '''
        # print('output_value_list[0]: ', type(output_value_list[0]))
        # if (filename == 'FUELS Training + Testing'):
        #     print('output_value_list[0]: ', type(list(output_value_list[0])))
        #     print('output_value_list[0]: ', output_value_list[0])
        #     for i in range(len(output_value_list)):
        #         if(output_value_list[i] is None)
        #         output_value_list[i] = list(output_value_list[i])
            
                
        ##To print just a tree structure with R2 value without "None" values uncomment
        ########################
        #    final result      #
        ########################



        ###For first node
        try:
            f.write('\\node {'+str(round(output_value_list[0],4))+'} %root\n')
        except TypeError: # ExceptioForCoefficient:
            f.write('\\node {'+str(output_value_list[0])+'} %root\n')
        # except ForFuelsTypes:


        ###for rest child
        skip_bracket_counter = 0
        for i in range(1,len_array):  #as starting from  1
            if(output_value_list[i] is not None ):
                try:    #train, test and data size....for those which have specific value 
                    f.write('child { node {'+ str(round(output_value_list[i],4))+'} ') 
                    # print('child { node {'+ str(round(output_value_list[i],4))+'} ') 
                except TypeError: #for coefficient 
                    f.write('child { node {'+ str(output_value_list[i])+'} ') 
                    # print('child { node {'+ str(output_value_list[i])+'} ') 
        



            if( i < len_array-type_of_division):
                # '''
                # if two keys are greater than current then do nothing 
                # if first is greater and second is less than one bracket 
                # if both less then two brackets
                # '''

                #if two values of index (i and i+1) is greater than new line 
                if(key_list_latex_array[i+1] > key_list_latex_array[i] and key_list_latex_array[i+2]  > key_list_latex_array[i]):
                    if(output_value_list[i] is not None):
                        f.write(' \n')
                        # print('\n')
                    # else:
                    #     f.write(' \n')

                #if only one value is greater (i+1) and (i+2) is less means another brach is going to come after one more child 
                elif(key_list_latex_array[i+1] > key_list_latex_array[i] and key_list_latex_array[i+2]  < key_list_latex_array[i]):
                    if(output_value_list[i] is not None):
                        f.write(' }\n')
                        # print(' }\n')
                    # else:
                    #     f.write(' \n')

                elif(key_list_latex_array[i+1] < key_list_latex_array[i] ):#and key_list_latex_array[i+2]  < key_list_latex_array[i]):
                    if(output_value_list[i] is not None):
                        f.write(' }\n')
                        # print(' }\n')
                    
                    j=i #index defined for another loop
                    end_bracket_counter = 0

                    # '''
                    # Pattern:
                    # Every closing bracket correspondence to the (division_type)^(index of closing bracket) in reverse direction
                    # here as binary 
                    # 2^(0) = 1
                    # 2^(1) = 2 :=2+1 =3                    
                    # 2^(2) = 4 :=3+4 =7
                    # so first closing bracket correspondence to 3rd data from last array then 2nd bracket realted to 7th position from last data 
                    # if array size is 100 : 
                    #     1st bracket 100-3  = 97
                    #     2nd bracket 100-7  = 93
                    #     2nd bracket 100-15 = 85
                    #     2nd bracket 100-31 = 69
                    #     2nd bracket 100-65 = 55

                    # '''

                    #calculating closing brackets
                    while(key_list_latex_array[j+1] < key_list_latex_array[i]):
                        end_bracket_counter += 1
                        j +=  1

                    # print('end_bracket_counter: ', end_bracket_counter)
                    #####calcualting how many actual bracket to print based on checking None data 
                    ##### if related data has None then reduce the bracket printing count 
                    print_closing_bracket = end_bracket_counter
                    location_relTo_bracket = i      #location related to closing bracket
                    for k in range(1,end_bracket_counter+1):
                        location_relTo_bracket -= (type_of_division)**(k)
                        # print(i ,':::: output_value_list[location_relTo_bracket]: ', output_value_list[location_relTo_bracket])
                        # print('location_relTo_bracket: ', location_relTo_bracket)
                        if(output_value_list[location_relTo_bracket] is None):  #if relavant location has None data then don't print bracket
                            print_closing_bracket -= 1
                        else:
                            continue

                    #printing brackets
                    for l in range(print_closing_bracket):
                        f.write(' }\n')
                        # print(' }\n')
                    continue
            #########################
            # add this based on type of divisions 
            #########################
            elif(i == len_array-3):
                if(type_of_division ==3):
                    if(output_value_list[i] is not None):
                        f.write('}\n')
                        # print('}\n')   
             
            elif(i == len_array-2):
                if(output_value_list[i] is not None):
                    f.write('}\n')
                    # print('}\n')             
                
                    continue
            elif(i == len_array-1):
                # print('output_value_list[i]: ', output_value_list[i])
                if(output_value_list[i] is not None):
                    f.write('}\n')
                location_relTo_bracket = i  #assining current location
                for m in range(1,self.number_of_levels-(type_of_division-1)):  #actually its -2 but as we have started from so 
                    location_relTo_bracket -= (type_of_division)**(m)
                    # print(i ,':::: output_value_list[location_relTo_bracket]: ', output_value_list[location_relTo_bracket])
                    # print('location_relTo_bracket: ', location_relTo_bracket)
                    if(output_value_list[location_relTo_bracket] is None):  #if relavant location has None data then don't print bracket
                        pass
                    else:
                        #printing brackets
                        f.write(' }\n')
                        
        ######################################
        # please set according to requirement#
        ######################################
        if(filename == 'Testing' or filename=='coefficient' or filename == 'max_rel_error' or filename == 'Training' or filename=='MaxRelError' or filename == 'Datasize' or filename == 'ChildLabel'):
            if(type_of_division == 3) :
                if(self.number_of_levels > 3):
                    for i in range(type_of_division-1):
                        f.write(' }\n')
                elif(self.number_of_levels == 3):
                    for i in range(type_of_division-2):
                        f.write(' }\n')
                    if(filename == 'Datasize' or filename == 'max_rel_error' or  filename == 'ChildLabel'):
                        f.write(' }\n')
                        
                elif(self.number_of_levels == 2 and self.cluster_label < 5):
                    for i in range(type_of_division-2):
                        f.write(' }\n')
                        
                elif(self.number_of_levels == 2 and self.cluster_label == 5):
                    if(filename == 'Datasize' or filename == 'max_rel_error' or  filename == 'ChildLabel'):
                        pass
                elif(self.number_of_levels == 1):
                    if(filename == 'Datasize' or filename == 'max_rel_error' or  filename == 'ChildLabel'):
                        f.write(' }\n')

        # for i in range(len(latex_supported_array)):
        f.write(';')                   
        f.write('\end{tikzpicture} \n')   
        f.write(" \caption{"+str(filename)+" plot } ")                             
        f.write(" \end{figure}\n")    
        f.write('\end{document} \n') 
        f.close()
        


if __name__ == "__main__":

    #Adding library 
    try:
            # '''
            # If  externally features are supplied given more prioritys
            # '''
            sys.path.append(self.curr_directory)
            from feature_selection import select_feature as Sel_feat
    except ImportError:
            from common.select_feature import select_feature as Sel_feat
        


    #Generating Dataset 
    Fuel_data = pd.read_csv('Alkane_Dataset_full.csv')
    criteria = 0.8
    path = '/home/pragnesh/Git/Data_driven_Kinetics/example'
    #calculating R2 value
    Flag_value = 8
    from common.data_gen import data_gen
    from tree.Tertiary_Tree import Ternary_Tree as TT
    from common.find_fuel_type import find_fuel_type 
    #finding out the straight chain alkanes
    list_fuel = find_fuel_type.find_strightchain_alkanes(Fuel_data)

    dataset = data_gen(Fuel_data,list_fuel,Flag_value,path)     #normal dataset generation
    df,tau = Sel_feat.feature_selection(dataset)
    Tree = Ternary_Tree(df,tau,.03,Flag_value,path)
    Tree.Implement_Tree()
