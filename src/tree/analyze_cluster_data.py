import pandas as pd
import numpy as np
import os
import subprocess

def analyze_cluster_data(curr_dir):
	path =str(curr_dir)+'/result/final_cluster/' 
	files = os.listdir(path)

	files_csv = [i for i in files if i.endswith('.csv')]
	files_csv.remove('full_data.csv')

	all_data = pd.read_csv(str(path)+'full_data.csv')
	for i in files_csv :
			cluster_i = pd.read_csv(str(path)+str(i))
			try:
				original_data_clu_i = all_data[all_data['Unnamed: 0'].isin(cluster_i['Unnamed: 0'])]
			except KeyError:
				continue
			original_data_clu_i.to_csv(str(path)+str(i),index=False)

	print('check  :',str(path), ' to check cluster data')
