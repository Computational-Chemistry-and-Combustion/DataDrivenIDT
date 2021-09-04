
##Directory to export the file of combination of different files
dir_path = './../'
import statsmodels.api as sm
import numpy as np
import pandas as pd
from result_check import result_check
import sys
import os
from search_fileNcreate import search_fileNcreate as SF

dir_path = os.path.dirname(os.path.realpath(__file__))
# print('dir_path: ', dir_path)
sys.path.append(dir_path)
import time

#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
    Main_folder_dir += dir_split[i] + str('/')

class Backward_elimination:

    # Backward Elimination with p-values only:
    def BackwardElimination_P(X_train,y_train,X_names,cluster_label,curr_directory,child_type='root',sl=0.05,elimination=False):
        
        '''
        Backpropagation is applied here to remove unnecessary feature based on ONlY p-value .
        Arguments : (X_train, sl,y_train,y_test,X_names)

        X_train : Feature vectors of dataset.
        X_test  : Feature vector of testing dataset which is obtained by equisized and remaining dataset 
        sl : Specify the limit of Accuracy. If the certain feature contains P-value more than sl value then 
             it is removed.     
        y_train: Actual result values of training data.
        y_test : Actual result values of testing data.
        X_names : X_names are headers of the dataset. Which is useful to show the feature in result table.  
        '''
        
        #Adding library 
        try:
            # '''
            # If  externally features are supplied given more prioritys
            # '''
            sys.path.append(curr_directory)
            from feature_selection import select_feature as Sel_feat
        except ImportError:
            from select_feature import select_feature as Sel_feat


        print('\n Statistical significance value for backward elimination is taken as ',str(sl),'or',str(100-float(sl)*100),'% confidence interval \n')
        headers = Sel_feat.column_selection()
        headers.append('training_adj_r2')
        headers = pd.Series(np.array(headers))
        
        
        numVars = X_train.shape[1]
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(y_train, X_train).fit()
            # print(regressor_OLS.summary())
            if(elimination is True):
                maxVar = max(regressor_OLS.pvalues)#.astype(float)
                if maxVar > sl:
                    for j in range(1, numVars - i): ###Starting with as don't want to reject Intercept 
                        if (regressor_OLS.pvalues[j].astype(float) == 'nan'): #if p value is nan reject 
                            try:
                                X_train = X_train.drop(X_train.columns[j],axis=1)  #pandas array passed
                            except AttributeError:
                                X_train = np.delete(X_train, j, 1)
                            ###if numpy array passes
                            # X_train = np.delete(X_train, j, 1)
                            # X_test = np.delete(X_test, j, 1)
                        if (regressor_OLS.pvalues[j].astype(float) == maxVar): #if p value is more than defined reject 
                            # print(X_train)
                            # print(j)
                            try:
                                X_train = X_train.drop(X_train.columns[j],axis=1)  #pandas array passed
                            except AttributeError:
                                X_train = np.delete(X_train, j, 1)
                            ###if numpy array passes
                            # X_train = np.delete(X_train, j, 1)
                            # X_test = np.delete(X_test, j, 1)
                            del X_names[j]                        
                            break
            else:
                break
    

        #####final_model after elimination for printing the result
        # result_check.results_print(regressor_OLS) #uncomment if you want to check result seperately

        training_adj_r2 = regressor_OLS.rsquared_adj
        summary = regressor_OLS.summary(xname=X_names)
        print('summary: ', summary)
        parameters = regressor_OLS.params
        parameter_name_dict = dict(zip(X_names,parameters))

        # #prediction result 
        # y_test_predicted = regressor_OLS.predict(X_test)

        # ###FINDING R2
        # ##It is for TESTING SET
        # from sklearn.metrics import r2_score
        # Testing_Adj_r2 = r2_score(y_test_predicted , y_test)
        ####File reading and saving result 
        SF.check_directory(str(curr_directory)+'/result/coefficients/'+str(child_type))
        SF.check_file_existence(str(curr_directory)+'/result/coefficients/'+str(child_type)+'/Result_Coefficients_'+str(cluster_label)+'.csv')
        try:
            #if file exist it will read and append the output 
            df =  pd.read_csv(str(curr_directory)+'/result/coefficients/'+str(child_type)+'/Result_Coefficients_'+str(cluster_label)+'.csv')    #reading dataset 
        except pd.errors.EmptyDataError:
            #if file doesn't exist it will create empty dataframe and append the output only with headers
            df = pd.DataFrame([],columns=headers)   #making dataframe with headers
            df = df[0:0]        #cleaning dataset 
            df = pd.DataFrame([],columns=headers)   #making dataframe with headers
            df.to_csv(str(curr_directory)+'/result/coefficients/'+str(child_type)+'/Result_Coefficients_'+str(cluster_label)+'.csv',index=False)    #saving dataframe
            df =  pd.read_csv(str(curr_directory)+'/result/coefficients/'+str(child_type)+'/Result_Coefficients_'+str(cluster_label)+'.csv')    #again reding csv to store result 
        # print('summary After Back elimination : \n', summary)
        coefficient_series = []        #to appended in dataset
        for i in range(len(headers)-1): #-1 as last entry later added ###training  r2
            if(headers.iloc[i] in X_names):
                coefficient_series.append(parameter_name_dict.get(headers.iloc[i]))   #Appending corresponding value
            else:
                coefficient_series.append(0)
        coefficient_series.append(training_adj_r2)
        # coefficient_series.append(Testing_Adj_r2)
        df1 = pd.DataFrame([coefficient_series],columns=headers)
        # coefficient_series = pd.Series(coefficient_series)
        df = df.append(df1)
        df.to_csv(str(curr_directory)+'/result/coefficients/'+str(child_type)+'/Result_Coefficients_'+str(cluster_label)+'.csv',index = False)    #saving dataframe
        # print(regressor_OLS.params)
        # print(X_names)
        return X_train, training_adj_r2, summary, X_names
        
        
    #Backward Elimination with p-values and Adjusted R Squared:
    def BackwardElimination_P_n_R(X_train, SL,y,X_names):
        '''
        Backpropagation is applied here to remove unnecessary feature based on p-value and Adjusted R^2 value.
        Arguments : (X_train,sl,y,X_names)

        X_train : Feature vectors of dataset.
        sl : Specify the limit of Accuracy. If the certain feature contains P-value more than sl value then 
             it is removed.     
        y : Acual result values of data.
        X_names : X_names are headers of the dataset. Which is useful to show the feature in result table.  
        '''
        x_rows = X_train.shape[0]
        x_cols = X_train.shape[1]
        numVars = len(X_train[0])
        temp = np.zeros((x_rows, x_cols)).astype(int)
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(y, X_train).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            adjR_before = regressor_OLS.rsquared_adj.astype(float)
            if maxVar > SL:
                for j in range(0, numVars - i):
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                        temp[:, j] = X_train[:, j]
                        X_train = np.delete(X_train, j, 1)
                        tmp_regressor = sm.OLS(y, X_train).fit()
                        adjR_after = tmp_regressor.rsquared_adj.astype(float)
                        if (adjR_before >= adjR_after):
                            x_rollback = np.hstack((X_train, temp[:, [0, j]]))
                            x_rollback = np.delete(x_rollback, j, 1)
                            # print (regressor_OLS.summary())
                            return x_rollback
                        

        adj_r2 = regressor_OLS.rsquared_adj
        summary = regressor_OLS.summary(xname=X_names)
        return X_train,adj_r2,summary
