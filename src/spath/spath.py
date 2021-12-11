 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import joblib
import copy
import sys
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge #change base with ridge to see change 
from sklearn.base import RegressorMixin, BaseEstimator, clone

from common.search_fileNcreate import search_fileNcreate as SF
from common.find_fuel_type import find_fuel_type 
from common.reference_point import reference_point
from common.select_feature import select_feature as Sel_feat
from common.data_gen import data_gen

#setting up paths
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)


#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
    Main_folder_dir += dir_split[i] + str('/')


class ClusteredRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_components=2, base=LinearRegression(),limited_ref_points=False, random_state=1, max_iter=100, tol=1e-10, verbose=False, data=[], curr_directory='.'):
        self.n_components = n_components #number of components
        self.base = base #base=ridge estimator
        self.random_state = random_state #random state for fix data
        self.max_iter = max_iter #maximum number of iteration for EM
        self.tol = tol #tolerance
        self.verbose = verbose # Save parameters with the same name.
        self.estimators_ = [clone(self.base) for i in range(self.n_components)] #cloning base estimator for all the estimator
        self.data = data #all the data of training 
        self.cluster_index = None
        self.curr_directory = curr_directory
        self.limited_ref_points = limited_ref_points

    def fit(self, X, y):#numpy random seed
        '''
        Initialized estimator let's say-3
        for each estimator assigned random weights
        goal: learn weights for each cluster

        '''
        
        np.random.seed(self.random_state) 

        # print('self.estimators_: ', self.estimators_)

        # initialize cluster responsibilities (weights) randomly 
        self.resp_ = np.random.uniform(size=(X.shape[0], self.n_components))
        # print('self.resp_: ', self.resp_)
        self.resp_ /= self.resp_.sum(axis=1, keepdims=True) #normallizing
        # print('self.resp_ : ', self.resp_ )

        for it in range(self.max_iter):# till maximum iteration
            old_resp = self.resp_.copy() #copy of responsibility as later it will get updated

            # Estimate sample-weithted regressions
            errors = np.empty(shape=self.resp_.shape) #empty error to store error

            #for each estimator fir model here-3
            for i, est in enumerate(self.estimators_): #for in estimators
                est.fit(X, y, sample_weight=self.resp_[:, i])
                errors[:, i] = y - est.predict(X) #storing error for each model
            # print('#######################')
            # print('errors: ', errors)
            # print('#######################')
            self.mse_ = np.sum(self.resp_ * errors**2) / X.shape[0] #calculating error for each model
            # print('self.mse_: ', self.mse_)

            if self.verbose:
                print(self.mse_)

            # Recalculate responsibilities
            # print('Recalculate responsibilities')
            self.resp_ = np.exp(-errors**2 / self.mse_)
            # print('self.resp_ : ', self.resp_ )
            self.resp_ /= self.resp_.sum(axis=1, keepdims=True)
            # print('self.resp_: ', self.resp_)

            # stop if small change in weights(responsibilities)
            delta = np.abs(self.resp_ - old_resp).mean()
            if delta < self.tol:
                break

        

        self.n_iter_ = it #updating number of iteration
        
        #asssign cluster to the data
        # self.cluster_index = np.array([np.argmin(errors[i], axis=0) for i in range(len(errors))])
        self.cluster_index = np.array(np.argmax(self.resp_, axis=1))

        self.storeObj_Result(X,y,self.curr_directory)
        return self
    
    def storeObj_Result(self,X,y,curr_directory):
        '''
        This method will store object and result
        '''
        # print('cluster_index: ', self.cluster_index)

        for i in range(self.n_components): #for each model
            # print(self.data)
            cluster_data = X[X.index.isin(np.where(self.cluster_index == i)[0])]
            y_act = y[y.index.isin(np.where(self.cluster_index == i)[0])]
            estimator_score = self.estimators_[i].score(cluster_data,y_act)
            pd.options.mode.chained_assignment = None #ignoring columns copies
            cluster_data['y_pred'] = self.estimators_[i].predict(cluster_data)
            cluster_data['y_act'] = y_act
            # print(cluster_data)

            #storing objects and ref points
            #finding and creating path
            SF.check_directory(str(curr_directory)+'/object_file/')

            #initialized object of ref points
            ref_point = reference_point(curr_directory,self.limited_ref_points)

            #calculating centroid
            centroid = ref_point.calculate_centroid(cluster_data) #pandas series
            SF.check_directory(str(curr_directory)+'/object_file/')
            SF.check_directory(str(curr_directory)+'/object_file/centroids/')

            ####object is stored
            filename_centroid =  str(curr_directory)+'/object_file/centroids/centroid_'+str(i)+'.sav'
            joblib.dump(centroid,filename_centroid)  
            
            # commented this part as it is required for final clusters and that we are going to
            # get after optimized cluster so no need to waste computation
            ref_point.other_reference_point(cluster_data,centroid,i,'cluster_ref')

            #final cluster gives stores only those cluster data which are useful fro prediction
            SF.check_directory(str(curr_directory)+'/result/final_cluster/')
            cluster_data.to_csv(str(curr_directory)+'/result/final_cluster/end_cluster_'+str(i)+'.csv')

            
            ##writing centroid
            ####For writing centroid
            SF.check_directory(str(curr_directory)+'/result/centroids/')
            cols = list(cluster_data.columns)
            try:
                centroid_headers = cols.remove('Constant')
            except ValueError:
                centroid_headers = cols

            try:
                centroid_out = pd.read_csv(str(curr_directory)+'/result/centroids/centroid_'+str(i)+'.csv')
            except pd.errors.EmptyDataError:
                centroid_out = pd.DataFrame([],columns=centroid_headers)
            except FileNotFoundError:
                centroid_out = pd.DataFrame([],columns=centroid_headers)
            centroid_out = centroid_out.append(pd.Series(centroid,index=centroid_headers),ignore_index=True)
            centroid_out.to_csv(str(curr_directory)+'/result/centroids/centroid_'+str(i)+'.csv',index=False)

            SF.check_directory(str(curr_directory)+'/object_file/regressor/')
            SF.check_directory(str(curr_directory)+'/object_file/x_names/')
            X_names = cols

            #saving object for further prediction
            filename_regressor = str(curr_directory)+'/object_file/regressor/regressor_'+str(i)+'.sav'
            joblib.dump(self.estimators_[i], filename_regressor)

            filename_xnames =  str(curr_directory)+'/object_file/x_names/xname_'+str(i)+'.sav'
            joblib.dump(X_names,filename_xnames)  

            SF.check_directory(str(curr_directory)+'/result/ID_comparison/')
            #checking directory 
            ID_comparison = pd.DataFrame()
            y_pred = cluster_data['y_pred']
            y_act = cluster_data['y_act']
            ID_comparison['y_predicted'] = y_pred
            ID_comparison['y_actual'] = y_act
            ID_comparison['Relative Error'] = np.abs(y_pred - y_act)/np.abs(y_act)
            ID_comparison.to_csv(str(curr_directory)+'/result/ID_comparison//ID_comparison_'+str(i)+'.csv')

            ####File reading and saving coefficient 
            SF.check_directory(str(curr_directory)+'/result/coefficients/')
            SF.check_file_existence(str(curr_directory)+'/result/coefficients/Result_Coefficients_'+str(i)+'.csv')
            headers = cols
            headers.insert(0,'Constant')
            headers.append('Train_R2')
            headers.remove('y_pred')
            headers.remove('y_act')
            # print(headers)

            try:
                #if file exist it will read and append the output 
                df =  pd.read_csv(str(curr_directory)+'/result/coefficients/Result_Coefficients_'+str(i)+'.csv')    #reading dataset 
            except pd.errors.EmptyDataError:
                #if file doesn't exist it will create empty dataframe and append the output only with headers
                df = pd.DataFrame([],columns=headers)   #making dataframe with headers
                df = df[0:0]        #cleaning dataset 
                df = pd.DataFrame([],columns=headers)   #making dataframe with headers
                df.to_csv(str(curr_directory)+'/result/coefficients/Result_Coefficients_'+str(i)+'.csv',index=False)    #saving dataframe
                df =  pd.read_csv(str(curr_directory)+'/result/coefficients/Result_Coefficients_'+str(i)+'.csv')    #again reding csv to store result 

                  #to appended in dataset
            coefficient_series = [i for i in self.estimators_[i].coef_]
            coefficient_series.insert(0,self.estimators_[i].intercept_)
            coefficient_series.append(estimator_score)
            # coefficient_series.append(Testing_Adj_r2)
            df1 = pd.DataFrame([coefficient_series],columns=headers)
            # coefficient_series = pd.Series(coefficient_series)
            df = df.append(df1)
            df.to_csv(str(curr_directory)+'/result/coefficients/Result_Coefficients_'+str(i)+'.csv',index = False)
    
    def dataBeforeProcessingClusterWise(self,dataset,curr_directory):
        for i in range(self.n_components): #for each model
            cluster_data = dataset[dataset.index.isin(np.where(self.cluster_index == i)[0])]
            #save data of all clusters after regression 
            SF.check_directory(str(curr_directory)+'/result/cluster_data/')
            cluster_data.to_csv(str(curr_directory)+'/result/cluster_data/'+'/cluster_'+str(i)+'.csv')

    def predict(self, X):
        """ Calculate a matrix of conditional predictions """
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
        return np.vstack([est.predict(X) for est in self.estimators_]).T #conditional prediction using all the cluster

    def predict_proba(self, X, y):
        """ Input X-test to make prediction and y for comparision
        Estimate cluster probabilities of labeled data """
        predictions = self.predict(X)
        errors = np.empty(shape=self.resp_.shape)
        for i, est in enumerate(self.estimators_):
            errors[:, i] = y - est.predict(X)
        resp_ = np.exp(-errors**2 / self.mse_) #based on least error assigning cluster
        resp_ /= resp_.sum(axis=1, keepdims=True)
        return resp_
    
    

# # use -f and -k flag
# def CycleTesting():
#     from common.data_gen import data_gen
#     external_data = pd.read_csv('FixedTest.csv')
#     try:
#         list_fuel = find_fuel_type.find_strightchain_alkanes(external_data)
#         dataset = data_gen(external_data,list_fuel,Flag_value,Path)     #normal feature generation
#     except KeyError:
#         pass

#     #old
#     from old_external_test_cycle import old_external_test_cycle
#     testset_obj_old = old_external_test_cycle(Flag_value,Path)
#     testset_obj_old.external_testset(dataset)

# def combineCluster():
#     from common.combined_N_analyze_all_test_result import combined_N_analyze_all_test_result
#     combined = combined_N_analyze_all_test_result(Path)
#     combined.process()
        




