'''
IF you have obtained coefficient many time by running the code many time.
The obtained coefficient will have variation due to train test set and 
variation in the data.
This code will return average value of coefficient obtained by finding out 
the average coefficient value and its histgram plot.
It will generate histogram plot for all the coefficients columns.
'''

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os 
import seaborn as sns
from search_fileNcreate import search_fileNcreate as SF
# import numpy as np

##Directory Capture
dir_path = os.path.dirname(os.path.realpath(__file__))
# print('dir_path: ', dir_path)
sys.path.append(dir_path)


#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
    Main_folder_dir += dir_split[i] + str('/')


class coefficient_plotting():
    def coefficient_mean_result_hist(df,curr_directory): 
        '''
        Find out the mean value from predicted coefficients and plot histogram
        read the given file 
        '''
    
        df = df.loc[:, (df != 0).any(axis=0)]   #removing colm contains only zeros 
        headers = list(df.columns)

        #Headers removed /
        for i,item in enumerate(headers):
            headers[i] = headers[i].replace('/','_')

        weights_mean = []
        # for j in range(500,df.shape[0],500): #by interval of 500
        for j in range(df.shape[0]-1,df.shape[0]): #all the data rows
            for i in range(df.shape[1]): #columnwise
                headers[i] = headers[i].replace('/','')
                data_col = df.iloc[:j,i] #till jth rows and ith column
                weights_mean.append(data_col.mean())

                #Histogram Plotting 
                # plt.rc('text', usetex=True)
                plt.clf()
                plt.hist(data_col)
                plt.xlabel('coefficient of '+str(headers[i]))
                if(i==0):
                    plt.xlabel('Intercept')
                plt.ylabel('Frequency')
                plt.axvline(data_col.mean(), color='k', linestyle='dashed', linewidth=1)
                _, max_ = plt.ylim()
                plt.text(data_col.mean(),max_/3,'Mean: {:.2f}'.format(data_col.mean()),horizontalalignment='center',verticalalignment='bottom',bbox=dict(facecolor='red', alpha=0.5))
                SF.check_directory(str(curr_directory)+'/coefficient_histogram_plots/')
                plt.savefig(str(curr_directory)+'/coefficient_histogram_plots/'+str(headers[i])+'.png')
                # plt.show()
                plt.close()
                

                ## another fit 
                # from astropy.modeling import models, fitting
                # bin_heights, bin_borders = np.histogram(data_col, bins='doane')
                # bin_widths = np.diff(bin_borders)
                # bin_centers = bin_borders[:-1] + bin_widths / 2

                # t_init = models.Gaussian1D()
                # fit_t = fitting.LevMarLSQFitter()
                # t = fit_t(t_init, bin_centers, bin_heights)

                # x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
                # plt.figure()
                # plt.bar(bin_centers, bin_heights, width=bin_widths, label='histogram')
                # plt.plot(x_interval_for_fit, t(x_interval_for_fit), label='fit', c='red')
                # plt.legend()
                # plt.show
                # plt.close()
        weights_mean_n_header = dict(zip(weights_mean, headers))

        avg_weights = list(weights_mean_n_header.keys())
        for i in range(len(headers)):
            print(avg_weights[i] , '\t', weights_mean_n_header.get(avg_weights[i]))
        return weights_mean_n_header

    def coefficient_mean_result_density(df,curr_directory): 
        '''
        Find out the mean value from predicted coefficients and plot histogram
        read the given file 
        '''
    
        df = df.loc[:, (df != 0).any(axis=0)]   #removing colm contains only zeros 
        headers = list(df.columns)

        #Headers removed /
        for i,item in enumerate(headers):
            headers[i] = headers[i].replace('/','_')

        weights_mean = []
        # for j in range(500,df.shape[0],500): #by interval of 500
        for j in range(df.shape[0]-1,df.shape[0]): #all the data rows
            for i in range(df.shape[1]): #columnwise
                headers[i] = headers[i].replace('/','')
                data_col = df.iloc[:j,i] #till jth rows and ith column
                weights_mean.append(data_col.mean())

                #Histogram Plotting 
                # plt.rc('text', usetex=True)
                plt.clf()
                # matplotlib histogram
                # plt.hist(data_col, color = 'blue', edgecolor = 'black')

                # seaborn histogram
                if(i==0):
                    label = 'Intercept'
                else:
                    label = str(headers[i])
                    print('label: ', label)
                    if(label == 'log_Fuel(%)'):
                        label = r'$\log(\chi_{fuel}$)'
                        
                    elif(label == 'log_Oxidizer(%)'):
                        label = r'$\log(\chi_{oxygen}$)'
                        
                    elif(label == 'log_P(atm)'):
                        label = r'$\log (P)$'
                        
                    elif(label == 'T0_S_H__T'):
                        label = r'$\frac{T_0}{{C_{SH}} \cdot T}$'
                        
                    elif(label == 'T0_T'):
                        label = r'$\frac{T_0}{T}$'
                        
                    elif(label == 'training_adj_r2'):
                        label = 'Adjusted $R^2$ (Training)'
                        

                sns.distplot(data_col, hist=True, kde=True, color = 'blue',hist_kws={'edgecolor':'black'})
                plt.rc('text', usetex=True)
                fontsize=19
                plt.xlabel('Range of '+str(label),fontsize=fontsize)
                plt.ylabel('Probability Density',fontsize=fontsize)
                plt.axvline(data_col.mean(), color='k', linestyle='dashed', linewidth=1)
                _, max_ = plt.ylim()
                plt.text(data_col.mean(),max_/3,'Mean: {:.2f}'.format(data_col.mean()),horizontalalignment='center',verticalalignment='bottom',bbox=dict(facecolor='red', alpha=0.5),fontsize=fontsize)
                SF.check_directory(str(curr_directory)+'/coefficient_histogram_plots/')
                plt.xticks(fontsize=fontsize, rotation=0)
                plt.yticks(fontsize=fontsize, rotation=0)
                plt.tight_layout()
                plt.savefig(str(curr_directory)+'/coefficient_histogram_plots/'+str(headers[i])+'.eps',orientation ='landscape',bbox_inches='tight')
                # plt.show()
                # plt.close()
                

                ## another fit 
                # from astropy.modeling import models, fitting
                # bin_heights, bin_borders = np.histogram(data_col, bins='doane')
                # bin_widths = np.diff(bin_borders)
                # bin_centers = bin_borders[:-1] + bin_widths / 2

                # t_init = models.Gaussian1D()
                # fit_t = fitting.LevMarLSQFitter()
                # t = fit_t(t_init, bin_centers, bin_heights)

                # x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
                # plt.figure()
                # plt.bar(bin_centers, bin_heights, width=bin_widths, label='histogram')
                # plt.plot(x_interval_for_fit, t(x_interval_for_fit), label='fit', c='red')
                # plt.legend()
                # plt.show
                # plt.close()
            # plt.savefig(str(curr_directory)+'/coefficient_histogram_plots/'+str(headers[i])+'.png')
        weights_mean_n_header = dict(zip(weights_mean, headers))

        avg_weights = list(weights_mean_n_header.keys())
        for i in range(len(headers)):
            print(avg_weights[i] , '\t', weights_mean_n_header.get(avg_weights[i]))
        return weights_mean_n_header
    
    def coefficient_mode_result_hist(df,curr_directory): 
        '''
        Find out the maximum occurrence value from predicted coefficients and plot histogram
        read the given file 
        '''
    
        df = df.loc[:, (df != 0).any(axis=0)]   #removing colm contains only zeros 
        headers = list(df.columns)

        #Headers removed /
        for i,item in enumerate(headers):
            headers[i] = headers[i].replace('/','_')

        weights_max = []
        for i in range(df.shape[1]):
            data_col = df.iloc[:,i]
            non_0_data = list(filter(lambda a: a != 0, list(data_col)))
            non_0_data = pd.Series(non_0_data)
            weights_max.append( non_0_data.mode()[0])

            #Histogram Plotting 
            # plt.rc('text', usetex=True)
            plt.clf()
            plt.hist(data_col)
            plt.xlabel('coefficient of '+str(headers[i]))
            if(i==0):
                plt.xlabel('Intercept')
            plt.ylabel('Frequency')
            plt.axvline(data_col.mean(), color='k', linestyle='dashed', linewidth=1)
            _, max_ = plt.ylim()
            
            plt.text(data_col.mean(),max_/3,'Mode: {:.2f}'.format(data_col.mean()),horizontalalignment='center',verticalalignment='bottom',bbox=dict(facecolor='red', alpha=0.5))
            SF.check_directory(str(curr_directory)+'/coefficient_histogram_plots/')
            plt.savefig(str(curr_directory)+'/coefficient_histogram_plots/'+str(headers[i])+'.png')            # plt.show()
            plt.close()
        weights_max_n_header = dict(zip(weights_max, headers))
        avg_weights = list(weights_max_n_header.keys())
        for i in range(len(headers)):
            print(avg_weights[i] , '\t', weights_max_n_header.get(avg_weights[i]))
        return weights_max_n_header
        
if __name__ == "__main__":
    # from coefficient_plotting import coefficient_plotting 
    #change_file_name
    filename = 'Result_Coefficients_2'
    df = pd.read_csv(filename+'.csv') #Reading file
    weights_result  = coefficient_plotting.coefficient_mean_result_density(df,filename)
     #############
    '''
    Uncomment this two part when want to analyse the effect of variation of sampaling on result so called
    boosting effect.
    Warning : keep in mind that in result folder *20000.csv result file is required
    # '''
    # print('Predicting Result Based on Previously Obtained result')
    # from Testset_prediction import Testset_prediction as TP
    # print('X_remaining_test: ', X_remaining_test)
    # weights_n_parameters = TP.trainingset_average_result() ##last value is R2
    # weights_mean = list(weights_n_parameters.keys())
    # weights_mean = weights_mean[:-1]
    # weights_mean  = np.array(weights_mean)
    # y_remaining_prediction = []
    # for i in range(len(X_remaining_test)):
    #     y_remaining_prediction.append(weights_mean.dot(X_remaining_test[i].T))
    # x_data = np.arange(len(X_remaining_test))
    # plt.figure()
    # plt.xlabel('Datapoints')
    # plt.ylabel('Ignition Delay')
    # plt.scatter(x_data,np.exp(y_remaining_prediction),marker = '*' ,c='red')
    # plt.scatter(x_data,(y_re_test),marker = '*',c='blue')
    # plt.show()


    # print('Predicitng Result Based on Previously Obtained result Using Max Probability')
    # from Testset_prediction import Testset_prediction as TP
    # print('X_remaining_test: ', X_remaining_test)
    # weights_max_n_parameters = TP.trainingset_max_result() ##last value is R2
    # weights_max = list(weights_max_n_parameters.keys())
    # weights_max = weights_max[:-1]
    # weights_max  = np.array(weights_max)
    # y_remaining_prediction = []
    # for i in range(len(X_remaining_test)):
    #     y_remaining_prediction.append(weights_max.dot(X_remaining_test[i].T))
    # x_data = np.arange(len(X_remaining_test))
    # plt.figure()
    # plt.xlabel('Datapoints')
    # plt.ylabel('Ignition Delay')
    # plt.scatter(x_data,np.exp(y_remaining_prediction),marker = '*' ,c='red')
    # plt.scatter(x_data,(y_re_test),marker = '*',c='blue')
    # plt.show()