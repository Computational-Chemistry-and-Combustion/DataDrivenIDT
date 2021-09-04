######################### External test File  ##############################################
# This  file will call external testset and based using saved object classify the data 
# into clusters using centroid and do regression on classified data


# NOTE : FILE IS SAME AS OLD_EXTERNAL_TEST just that IT CAN STORE ALL TEST RESULT SPERATELY


#############################################################################################

import numpy as np
import pandas as pd 
import joblib
 ###Heat Map###
# import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import copy
import warnings
import subprocess
##Directory to export the file of combination of different files
dir_path = './../'
import sys
import os 
import seaborn as sns
from search_fileNcreate import search_fileNcreate as SF


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path =  dir_path+'/analysis/Analysis_clusters_3'

# print('dir_path: ', dir_path)
sys.path.append(dir_path)
#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
	Main_folder_dir += dir_split[i] + str('/')


class combined_N_analyze_all_test_result():

		'''
		Checks new fuel which are not part of training or testing set
		'''
		def __init__(self,curr_directory):
			self.curr_directory = curr_directory
			self.result_dir = curr_directory+'/all_test_result'

		def combine_Files(self):
			'''
			In specified directory -- it will combine all files in sub-directory and 
			will generate final file 'merged_result.csv' in all sub direcory
			'''
			folders = os.listdir(str(self.result_dir))
			print(folders)
			for i in folders:
				dataset = pd.DataFrame([])
				file_list = os.listdir(str(self.result_dir)+'/'+str(i))
				for j in file_list:
					df = pd.read_csv(str(self.result_dir)+'/'+str(i)+'/'+str(j),index_col=False)
					if dataset.empty:
						dataset = df
					else:
						dataset = pd.concat([dataset, df], ignore_index=True)
				dataset.drop(dataset.filter(regex="Unnamed"),axis=1, inplace=True)
				dataset.to_csv(str(self.result_dir)+'/'+str(i)+'/merge_dataset.csv',index=False)

		def merged_all_file(self):
			'''
			This method will merge all 'merged_result.csv' files to compare result
			'''
			folders = os.listdir(str(self.result_dir))
			dataset = pd.DataFrame([])
			count = 1
			for i in folders:
				df = pd.read_csv(str(self.result_dir)+'/'+str(i)+'/merge_dataset.csv',index_col=False)
				df = df.sort_values('log_P(atm)')
				df = df.reset_index(drop=True)
				if dataset.empty:
					
					dataset = df
					print(dataset)
				else:
					dataset = dataset.sort_values('log_P(atm)')
					dataset = dataset.reset_index(drop=True)
					dataset['Time(μs)_predicted_'+str(count)] = df['Time(μs)_predicted']
					count += 1
					print(dataset.head())
			dataset.to_csv(str(self.result_dir)+'/final.csv',index=False)
			print(dataset.shape)

		def plot_Hist(self):
			'''
			with mean bar value
			'''
			df = pd.read_csv(str(self.result_dir)+'/final.csv')
			for i in range(len(df)):
				actual = df.iloc[i,7]
				data = df.iloc[i,8:]
				data_mean = sum(data)/len(data)
				sns.distplot(data, hist=True, kde=True, color = 'blue',hist_kws={'edgecolor':'black'})
				plt.xlabel('Variation in prediction of Ingition delay')
				plt.ylabel('Probability Density')
				plt.axvline(data_mean, color='k', linestyle='dashed', linewidth=1)
				_, max_ = plt.ylim()
				plt.text(data_mean,max_/2,'Mean : {:.2f}'.format(data_mean),horizontalalignment='center',verticalalignment='bottom',bbox=dict(facecolor='red', alpha=0.5))
				SF.check_directory(str(self.curr_directory)+'/prediction_plots/')
				plt.legend(['Actual value : {:.2f}'.format(actual)],loc='upper right')
				plt.savefig(str(self.curr_directory)+'/prediction_plots/point_'+str(i+1)+'.png')
				plt.close()

				error = [i - actual for i in data]
				error_mean = sum(error)/len(error)
				sns.distplot(error, hist=True, kde=True, color = 'blue',hist_kws={'edgecolor':'black'})
				plt.xlabel('Variation in prediction error of Ingition delay')
				plt.ylabel('Probability Density')
				plt.axvline(error_mean, color='k', linestyle='dashed', linewidth=1)
				_, max_ = plt.ylim()
				plt.text(error_mean,max_/2,'Error Mean : {:.2f}'.format(error_mean),horizontalalignment='center',verticalalignment='bottom',bbox=dict(facecolor='red', alpha=0.5))
				SF.check_directory(str(self.curr_directory)+'/error_prediction_plots/')
				plt.savefig(str(self.curr_directory)+'/error_prediction_plots/point_'+str(i+1)+'.png')
				plt.close()

		def plot_Hist_max(self):
			'''
			with most frquent bar value
			'''
			df = pd.read_csv(str(self.result_dir)+'/final.csv')
			bins = 70
			fontsize=19
			for i in range(len(df)):
			        actual = df.iloc[i,7]
			        data = df.iloc[i,8:]
			        hist, bin_edges = np.histogram(data, bins=bins)
			        data_max_frquent = bin_edges[np.argmax(hist)]
			        plt.rc('text', usetex=True)
			        sns.distplot(data, hist=True, kde=True, color = 'blue',hist_kws={'edgecolor':'black'},bins=bins)
			        plt.xlabel('Variation in prediction of ignition delay',fontsize=fontsize)
			        plt.ylabel('Probability Density',fontsize=fontsize,)
			        plt.axvline(data_max_frquent, color='k', linestyle='dashed', linewidth=1)
			        _, max_ = plt.ylim()
			        plt.xlim([4.5, 9.00])
			        plt.text(data_max_frquent,max_/2,'Most frequent : {:.2f}'.format(data_max_frquent),horizontalalignment='center',verticalalignment='bottom',bbox=dict(facecolor='red', alpha=0.5),fontsize=fontsize)
			        plt.xticks(fontsize=fontsize, rotation=0)
			        plt.yticks(fontsize=fontsize, rotation=0)
			        SF.check_directory(str(self.curr_directory)+'/prediction_plots/')
			        plt.legend(['Actual value : {:.2f}'.format(actual)],loc='upper right',fontsize=fontsize)
			        plt.tight_layout()
			        plt.savefig(str(self.curr_directory)+'/prediction_plots/point_'+str(i+1)+'.eps',orientation ='landscape')
			        plt.close()

			        error = [i - actual for i in data]
			        hist, bin_edges = np.histogram(error, bins=bins)
			        error_max_frequent = bin_edges[np.argmax(hist)]
			        plt.rc('text', usetex=True)
			        sns.distplot(error, hist=True, kde=True, color = 'blue',hist_kws={'edgecolor':'black'},bins=bins)
			        plt.xlabel('Variation in prediction error of ignition delay',fontsize=fontsize)
			        plt.ylabel('Probability Density',fontsize=fontsize)
			        plt.axvline(error_max_frequent, color='k', linestyle='dashed', linewidth=1)
			        _, max_ = plt.ylim()
			        plt.text(error_max_frequent,max_/2,'Most frequent error : {:.2f}'.format(error_max_frequent),horizontalalignment='center',verticalalignment='bottom',bbox=dict(facecolor='red', alpha=0.5),fontsize=fontsize)
			        plt.xticks(fontsize=fontsize, rotation=0)
			        plt.yticks(fontsize=fontsize, rotation=0)
			        plt.tight_layout()
			        SF.check_directory(str(self.curr_directory)+'/error_prediction_plots/')
			        plt.savefig(str(self.curr_directory)+'/error_prediction_plots/point_'+str(i+1)+'.eps',orientation ='landscape')
			        plt.close()
    

		def process(self,):
			'''
			Comment out fitst two lines if you want to just plot the graphs 
			and already have results
			# '''
			# self.combine_Files()
			# self.merged_all_file()
			# self.plot_Hist()
			self.plot_Hist_max()
			
			
if __name__ == "__main__":
	combined = combined_N_analyze_all_test_result(dir_path)
	combined.process()