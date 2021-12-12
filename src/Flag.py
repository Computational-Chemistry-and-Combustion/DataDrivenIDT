##############################################################
#              To take input of your Flag                  #
##############################################################

#Note: curre_dir : /home/pragnesh/Git/Data_driven_Kinetics/CleanCode/testing_place 
#does not contain / in the end 


import sys
import os 
import pandas as pd
import numpy as np
from common.find_fuel_type import find_fuel_type 
from common.search_fileNcreate import search_fileNcreate as SF
dir_path = os.path.dirname(os.path.realpath(__file__))
# print('dir_path: ', dir_path)
sys.path.append(dir_path)
# sys.path.append(dir_path+str('/clustering_methods'))


#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
    Main_folder_dir += dir_split[i] + str('/')

    
    
class Flag():
    '''
    This class method for Flags and diverts the code accordingly 
    '''
    def switch_func(Flag_value,method='common',process=None,dataset_location=None,curr_directory=None,smile=None,
                    division_error_criteria=0.05,elimination=False,sl=0.05,limited_ref_points=False,
                    n_cluster='2',max_iter=100,analysis_type=None):
        '''
        This methos works like switch in c++.
        According to your Flag it will divert the code and 
        process the code given Flag values.
        Argument Passed :(value)
        
        value is digit of your Flag dislayed in Flag method.
        Here, value is coming from Flag method.
        '''
        #Adding library 
        try:
            # '''
            # If  externally features are supplied given more priority
            # '''
            sys.path.append(curr_directory)
            from feature_selection import select_feature as Sel_feat
        except ImportError:
            from select_feature import select_feature as Sel_feat
        try:
            n_cluster_list = n_cluster.split('-')
            min_cluster=int(n_cluster_list[0])
            max_cluster=int(n_cluster_list[1])
        except IndexError:
            max_cluster=int(n_cluster)

        #Calling dataset 
        if (method == 'common'):

            if (Flag_value == '-a'):
                if(analysis_type=='manual'):
                    #importing library 
                    print('Analyzing data')
                    from common.Data_analysis import Data_analysis 
                    Data_analysis.View_n_Analyze(dataset_location,curr_directory) #calling method to generate analyze the dataset 
                    print('\n\n Check directory ./result/data_analysis/')
                if(analysis_type=='distribution'):
                    from common.featureAnalyze.ClusterDataAnalysis import ClusterDataAnalysis
                    ClusterDataAnalysis(curr_dir=curr_directory,dataset_location=dataset_location)
                    
                if(analysis_type=='parameter'):
                    from common.featureAnalyze.clusterParamaterAnalysis import clusterParamaterAnalysis
                    clusterParamaterAnalysis(curr_dir=curr_directory,dataset_location=dataset_location)
                    
                if(analysis_type=='pressure'):
                    from common.featureAnalyze.PressureAnalysis import PressureAnalysis
                    PressureAnalysis(curr_dir=curr_directory,dataset_location=dataset_location)
                    
                if(analysis_type=='temperature'):
                    from common.featureAnalyze.tempAnalysis import tempAnalysis
                    tempAnalysis(curr_dir=curr_directory,dataset_location=dataset_location)
                    

            elif (Flag_value == '-b'):
                # '''
                # Smile base bond information 
                # '''
                print('Finding the bond details')
                #importing library 
                from common.Bond_Extraction import Bond_Extraction as BE
                smile_input = [str(smile)]	#Argument List 
                Bond_details = BE.Bond_Extract(smile_input,curr_directory)
                print(Bond_details)
                SF.check_directory(str(curr_directory+'/result/Bond_details/'))
                SF.check_file_existence(str(curr_directory)+'/result/Bond_details/SMILE_result.csv')
                #reading old bond details
                try:
                    Bond_dataframe = pd.read_csv(str(curr_directory)+'/result/Bond_details/SMILE_result.csv')
                    Bond_details = pd.concat([Bond_dataframe,Bond_details]) #Concatenating two dataframes
                except pd.errors.EmptyDataError:
                    pass
                #appending new data
                Bond_details.to_csv(str(curr_directory)+'/result/Bond_details/SMILE_result.csv',index=False)
                
                print('\n\n Check directory ./result/Bond_details/')

            elif(Flag_value == '-h'):
                # '''
                # Fuel analysis 
                # This flag will analyze the fuel data and generate histogram plots of properties based on different fuels
                # '''
                print("## You are going to proceed for All straight chain alkanes fuel available in the dataset ## \n")
                Fuel_data = pd.read_csv(dataset_location)
                from common.fuel_analysis import fuel_analysis #calling fuela analysis part
                fuel_analysis.fuel_data_analysis(Fuel_data,curr_directory)     #passing thw whole dataset to it 

            elif(Flag_value == '-d'):
                    # '''
                    # gives Transformed data
                    # '''
                    from common.data_gen import data_gen
                    external_data = pd.read_csv(dataset_location)
                    list_fuel = find_fuel_type.find_strightchain_alkanes(external_data)
                    dataset = data_gen(external_data,list_fuel,Flag_value,curr_directory)     #normal dataset generation
                    df,tau = Sel_feat.feature_selection(dataset)
                    df['Time'] = tau
                    df.to_csv(str(curr_directory)+'/tranformed_data.csv',index=False)

            elif(Flag_value == '-k'):
                    # '''
                    # External test-cases
                    # By this flag can be used to store all the prediction result
                    # '''
                    from common.data_gen import data_gen
                    external_data = pd.read_csv(dataset_location)
                    list_fuel = find_fuel_type.find_strightchain_alkanes(external_data)
                    dataset = data_gen(external_data,list_fuel,Flag_value,curr_directory)     #normal feature generation

                    #old
                    from common.old_external_test_cycle import old_external_test_cycle
                    testset_obj_old = old_external_test_cycle(Flag_value,curr_directory)
                    testset_obj_old.external_testset(dataset)

            elif(Flag_value == '-f'):

                    from common.combined_N_analyze_all_test_result import combined_N_analyze_all_test_result
                    combined = combined_N_analyze_all_test_result(curr_directory)
                    combined.process()

                    
            elif(Flag_value == '-p'):
                    # '''
                    # Plot of average coefficient value obtained by histogram of coefficients
                    # '''
                    coef_data = pd.read_csv(dataset_location)       
                    from common.coefficient_plotting import coefficient_plotting as CP 
                    weights_mean_n_header = CP.coefficient_mean_result_density(coef_data,curr_directory)
                    print('\n\n Executed Normally! Please check plot Folder')

        if(method == 'GMM'):
            if(process == 'train'):
                
                try:
                    #finding out the straight chain alkanes
                    Fuel_data = pd.read_csv(dataset_location)
                    list_fuel = find_fuel_type.find_strightchain_alkanes(Fuel_data)

                    from common.data_gen import data_gen
                    dataset = data_gen(Fuel_data,list_fuel,Flag_value, curr_directory)     #normal dataset generation
                    SF.check_directory(str(curr_directory)+'/data/')
                    dataset.to_csv(str(curr_directory)+'/data/full_data.csv')
                except KeyError:
                    dataset = pd.read_csv(dataset_location)
                    
                #feature sepeation and selection
                df,tau = Sel_feat.feature_selection(dataset)
                df.to_csv(str(curr_directory)+'/Transformed.csv',index=False)

                from sklearn.mixture import GaussianMixture 
                from gmm.gmm import seperateClusterNregression
                aic_list =[]
                bic_list = []
                score_list = []

                if('-' in n_cluster):
                    for i in range(min_cluster,max_cluster+1):
                        gm = GaussianMixture(n_components=i, random_state=0).fit(df,tau)
                        out_directory = curr_directory+'/Analysis_'+str(i)
                        clusterIndex = gm.predict(df)
                        aic_list.append(gm.aic(df))
                        bic_list.append(gm.bic(df))
                        score_list.append(gm.score(df))
                        seperateClusterNregression(df,clusterIndex,tau,i,Fuel_data,out_directory,limited_ref_points)

                    import matplotlib.pyplot as plt

                    print('\n\n Executed Normally! Please check plot Folder')
                    plt.rc('text', usetex=True)
                    fontsize=19
                    index = [i for i in range(min_cluster,max_cluster+1)]
                    plt.plot(index,aic_list,'r-',label='AIC')
                    plt.plot(index,bic_list,'g-',label='BIC')
                    plt.title('GMM - AIC \& BIC criterion',fontsize=fontsize)
                    plt.xticks(index,fontsize=fontsize)
                    plt.yticks(fontsize=fontsize)
                    plt.xlabel('Number of clusters',fontsize=fontsize)
                    plt.ylabel('AIC/BIC',fontsize=fontsize)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(curr_directory+'/'+'AIC_BIC.eps',dpi=600,orientation ='landscape')

                    plt.plot(score_list,'b-',label='log-likelihood criterion')
                    plt.title('GMM - log-likelihood criterion',fontsize=fontsize)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(curr_directory+'/'+'log_likelihood.eps',dpi=600,orientation ='landscape')

                else:
                    gm = GaussianMixture(n_components=max_cluster, random_state=0).fit(df,tau)
                    clusterIndex = gm.predict(df)
                    aic_list.append(gm.aic(df))
                    bic_list.append(gm.bic(df))
                    score_list.append(gm.score(df))
                    seperateClusterNregression(df,clusterIndex,tau,max_cluster,Fuel_data,curr_directory,limited_ref_points)
                print('\n\n Executed Normally! Please check plot Folder')

            if(process == 'test'):
                from common.data_gen import data_gen
                external_data = pd.read_csv(dataset_location)
                try: #if fuel data passed try this else skip and prcocess
                    list_fuel = find_fuel_type.find_strightchain_alkanes(external_data)
                    external_data = data_gen(external_data,list_fuel,Flag_value,curr_directory)     #normal dataset generation
                except KeyError:
                    pass

                #old
                from common.old_external_test import old_external_test
                testset_obj_old = old_external_test(Flag_value,curr_directory)
                testset_obj_old.external_testset(external_data)


        if(method == 'multi'):
            if(process == 'train'):
                print("## You are going to proceed for All straight chain alkanes fuel available in the dataset ## \n")
                try:
                    #finding out the straight chain alkanes
                    from common.data_gen import data_gen
                    Fuel_data = pd.read_csv(dataset_location)

                    list_fuel = find_fuel_type.find_strightchain_alkanes(Fuel_data)

                    dataset = data_gen(Fuel_data,list_fuel,Flag_value,curr_directory)     #normal dataset generation
                except KeyError:
                    dataset = pd.read_csv(dataset_location)
                
                df,tau = Sel_feat.feature_selection(dataset)
                df.to_csv(str(curr_directory)+'/Transformed.csv',index=False)
                
                # #######################
                # # generate_data_points#
                # #######################
                # from multiple_regression.generate_data_points import generate_data_points as GDP
                # extended_dataset = GDP.generate_dataset(datastet,Flag_value)

                # ###########
                # # equisize#
                # ###########
                # from equilize_dataset import equilize_dataset
                # equisize_dataset , Diff_dataset   = equilize_dataset(datastet,Flag_value) 
                # #equisized dataset returns,
                # #equisized_dataset -- have equal datapoints of all fuel w.r.t to least data available for any fuel 
                # #Diff_datset -- the remaining data points after equi-sizing tha dataset  --- use as testing
                
                from multiple_regression.regression import regression as reg
                max_relative_error_training,training_adj_r2,Testing_Adj_r2,summary,coefficient_dictionary,dataset = reg(df,tau,curr_directory,elimination=elimination,sl=sl)
                print('\n\n Executed Normally! ')

            elif(process == 'test'):
                    # '''
                    # External test-cases
                    # '''
                    from common.data_gen import data_gen
                    external_data = pd.read_csv(dataset_location)
                    try: #if fuel data passed try this else skip and prcocess
                        list_fuel = find_fuel_type.find_strightchain_alkanes(external_data)
                        external_data = data_gen(external_data,list_fuel,Flag_value,curr_directory)     #normal dataset generation
                    except KeyError:
                        pass

                    #old
                    from common.old_external_test import old_external_test
                    testset_obj_old = old_external_test(Flag_value,curr_directory)
                    testset_obj_old.external_testset(external_data)
        
        if(method == 'spath'):
            if(process == 'train'):

                #finding out the straight chain alkanes
                from common.data_gen import data_gen
                import copy 

                try:
                    Fuel_data = pd.read_csv(dataset_location)
                    list_fuel = find_fuel_type.find_strightchain_alkanes(Fuel_data)
                    dataset = data_gen(Fuel_data,list_fuel,Flag_value, curr_directory)     #normal dataset generation
                    
                    SF.check_directory(str(curr_directory)+'/data/')
                    dataset.to_csv(str(curr_directory)+'/data/full_data.csv')

                    #feature sepeation and selection
                    df,tau = Sel_feat.feature_selection(dataset)

                    transformedData = copy.deepcopy(df)
                    transformedData['Time(μs)'] = tau
                except KeyError:
                    dataset = pd.read_csv(dataset_location)
                    df,tau = Sel_feat.feature_selection(dataset)

                    transformedData = copy.deepcopy(df)
                    transformedData['DependentVari'] = tau
                

                #training
                from spath.spath import ClusteredRegressor
                from sklearn.linear_model import LinearRegression, Ridge #change base with ridge to see change 

                model = ClusteredRegressor(n_components=max_cluster, base=LinearRegression(), limited_ref_points=limited_ref_points,random_state=1, max_iter=100, tol=1e-10, verbose=False, data=transformedData, curr_directory=curr_directory)
                model.fit(df, tau)
                labels = np.argmax(model.resp_, axis=1)

                model.dataBeforeProcessingClusterWise(dataset,curr_directory)

            if(process == 'test'):
                from common.data_gen import data_gen
                external_data = pd.read_csv(dataset_location)
                try: #if fuel data passed try this else skip and prcocess
                    list_fuel = find_fuel_type.find_strightchain_alkanes(external_data)
                    external_data = data_gen(external_data,list_fuel,Flag_value,curr_directory)     #normal dataset generation
                except KeyError:
                    pass

                #old
                from common.old_external_test import old_external_test
                testset_obj_old = old_external_test(Flag_value,curr_directory)
                testset_obj_old.external_testset(external_data)

        if(method == 'tree'):
            if(process == 'train'):
                    # '''
                    # This Flag is same as three but before transferring the data to find out R2,
                    # Data has to be transferred to tree structure and divide the data in middle way.
                    # Add tree module without uncertainty
                    # '''

                    print("## Tree Structure and data division for alkanes only## \n")

                    from common.data_gen import data_gen
                    from tree.Ternary_Tree import Ternary_Tree as TT
                    from multiple_regression.combine_clusters import combine_clusters as CC

                    try:
                        Fuel_data = pd.read_csv(dataset_location)
                        #finding out the straight chain alkanes
                        list_fuel = find_fuel_type.find_strightchain_alkanes(Fuel_data)
                        dataset = data_gen(Fuel_data,list_fuel,Flag_value,curr_directory)     #normal dataset generation
                        SF.check_directory(str(curr_directory)+'/result/final_cluster/')
                        dataset.to_csv(str(curr_directory)+'/result/final_cluster/full_data.csv')
                    except KeyError:
                        dataset = pd.read_csv(dataset_location)

                    df,tau = Sel_feat.feature_selection(dataset)
                    df.to_csv(str(curr_directory)+'/Transformed.csv')
                    Tree = TT(df,tau,division_error_criteria,Flag_value,curr_directory,elimination=elimination,sl=sl,limited_ref_points=limited_ref_points)
                    Tree.Implement_Tree()
                    
                    try:
                        # generating original cluster wise data 
                        from tree.analyze_cluster_data import analyze_cluster_data
                        analyze_cluster_data(curr_dir=curr_directory)
                    except ValueError:
                        pass

                    # #Training Result Analyzer
                    # print('\n\n\ntraining_accuracy')
                    # from tree.training_accuracy_check import training_accuracy_check
                    # train_accu = training_accuracy_check(Flag_value,curr_directory)
                    # train_accu.training_accuracy(dataset)


                    # #optimizing cluster
                    # final_clusters = CC(curr_directory,division_error_criteria,Flag_value)
                    # final_clusters.optimize_cluster()

                    print('\n\n Executed Normally! Please check plot Folder')
                    # os.system('sh ./for_ploting.sh')

            elif(process == 'test'):
                    # '''
                    # External test-cases
                    # '''
                    from common.data_gen import data_gen
                    external_data = pd.read_csv(dataset_location)
                    try: #if fuel data passed try this else skip and prcocess
                        list_fuel = find_fuel_type.find_strightchain_alkanes(external_data)
                        external_data = data_gen(external_data,list_fuel,Flag_value,curr_directory)     #normal dataset generation
                    except KeyError:
                        pass

                    #old
                    from common.old_external_test import old_external_test
                    testset_obj_old = old_external_test(Flag_value,curr_directory)
                    testset_obj_old.external_testset(external_data)

                    # # new
                    # from external_test import external_test 
                    # testset_obj = external_test(Flag_value,curr_directory)
                    # testset_obj.external_testset(external_data)
                    # print('\n\n Executed Normally! Please check plot Folder')
