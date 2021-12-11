import numpy as np
import matplotlib.pyplot as plt
import joblib
import subprocess
import pandas as pd
import time
from regression import regression
from search_fileNcreate import search_fileNcreate as SF
import copy
import sys 
from reference_point import reference_point

class combine_clusters():
	'''
	This module will try combine all the clusters and optimize the clusters
	each nodes contains following plan,
	self.data = data    # data
	self.y_dependent = y #dependent variable 
	self.r2 = None      #training R2 value 
	self.testing_r2 = None
	self.coefficients_dictionary = None
	self.data_size = None
	self.max_relerr_train = None
	self.centroid = None
	'''		
	def __init__(self,curr_directory,criteria,choice,process_type='optimized_cluster'):
		self.choice =choice
		self.curr_directory = curr_directory
		self.criteria = criteria
		self.final_cluster_count = 0
		self.ref_point = reference_point(self.curr_directory)
		self.process_type = process_type

		#update optimization
		self.optimized_nodes = []
		self.optimized_regressor = []
		self.optimized_file_name_label = []


	def find_total_clusters(self,directory_path):
		'''
		It will find number of files in gives path 
		'''
		#counting number files in the centroid directory which is total number of centroids
		cmd_num_of_files = "find "+directory_path+" -type f | wc -l"
		#check_output return output of bash 
		num_of_cluster = int(subprocess.check_output(cmd_num_of_files,shell=True, universal_newlines=False))  # returns the exit code in unix

		#finding name of files in the centroid directory
		cmd_files_name = "ls "+directory_path
		try:
			centroid_file_names = str(subprocess.check_output(cmd_files_name,shell=True, universal_newlines=False),"utf-8").split('\n') #converting output into string and then splitting 
		except subprocess.CalledProcessError:
			print('\n\n One cluster exist so nothing to optimize')
			sys.exit(0)
		file_names = [] #storing  file names
		file_name_label = []
		for i in range(len(centroid_file_names)-1):
				file_names.append(centroid_file_names[i]) #storing file labels
				# file_name_label.append(centroid_file_names[i][:-4].split('_')[-2]+'_'+centroid_file_names[i][:-4].split('_')[-1]) #storing centroid index -- cluster label #useful for other file reading
				file_name_label.append(int(centroid_file_names[i][:-4].split('_')[-1])) #storing centroid index -- cluster label #useful for other file reading
		file_dict = dict(zip(file_name_label,file_names))
		file_name_label=sorted(file_dict.keys())
		file_name_sorted = []
		for i,item in enumerate(file_name_label):
			file_name_sorted.append(file_dict.get(file_name_label[i]))

		# file_names = file_names.sort()
		return num_of_cluster,file_name_sorted, file_name_label


	def save_object_file(self,nodes,regressor,cluster_label):
		'''
		If traversing over list and checking nodes if final node is encountered
		and nothing left to see further then first node will be saved and purged 
		'''
		# print('nodes: ',len(nodes))
		for i,item in enumerate(nodes):
			#dumping centroid
			filename =  str(self.curr_directory)+'/object_file/final_centroid/centroid_'+str(cluster_label[i])+'.sav'
			centroid = self.ref_point.calculate_centroid(nodes[i].data)
			joblib.dump(centroid,filename) 

			#dumping other points
			self.ref_point.other_reference_point(nodes[i].data,centroid,cluster_label[i])

			#dumping regressor
			filename =  str(self.curr_directory)+'/object_file/final_regressor/regressor_'+str(cluster_label[i])+'.sav'
			joblib.dump(regressor[i],filename) 


			###storing result 
			#final cluster gives stores only those cluster data which are useful fro prediction
			SF.check_directory(str(self.curr_directory)+'/result/final_cluster_result/')
			nodes[i].data.to_csv(str(self.curr_directory)+'/result/final_cluster_result/end_cluster_'+str(cluster_label[i])+'.csv')
		

	def generate_regressor(self,nodes,data,cluster_label, node_index, choice='-o'):
		'''
		updating nodes paramater by regression 
		'''
		y = data['y_act']
		data = data.drop(columns=['y_act'])
		data = data.drop(columns=['y_pred'])
		# '''
		# This method will generate regressor
		# '''
		#again doing regression
		max_relative_error_training,training_adj_r2,Testing_Adj_r2,summary,coefficient_dictionary,dataset = regression(data,y,self.curr_directory,cluster_label=cluster_label,elimination=False,child_type='root', process_type=self.process_type)
		print('dataset: ', dataset.shape)
		print('max_relative_error_training: ', max_relative_error_training)

		#centroid 
		centroid = self.ref_point.calculate_centroid(dataset)

		print(str(self.curr_directory)+'/object_file/regressor/regressor_'+str(cluster_label)+'.sav')
		#loading regressor
		regre = joblib.load(str(self.curr_directory)+'/object_file/regressor/regressor_'+str(cluster_label)+'.sav')

		#updating relevant paramaters
		nodes[node_index].data = dataset    # data
		nodes[node_index].y_dependent = dataset['y_act'] #dependent variable 

		# ''' Nodes are updated here '''
		# nodes[node_index].left_node = None  #object left child
		# nodes[node_index].right_node = None  #object right child
		# nodes[node_index].center_node = None
		nodes[node_index].child_label = cluster_label
		nodes[node_index].r2 = training_adj_r2      #training R2 value 
		nodes[node_index].testing_r2 = Testing_Adj_r2
		nodes[node_index].coefficients_dictionary = coefficient_dictionary
		nodes[node_index].data_size = dataset.shape[0]
		# nodes[node_index].uniq_fuel = None
		nodes[node_index].max_relerr_train = max_relative_error_training
		nodes[node_index].centroid = centroid

		return nodes,regre
	
	def sort_by_data_size(self,load_nodes,load_regressor,file_name_label):
		'''
		It will sort regressor and node objects based on data-size associated with it
		'''
		regressor = []
		data_size = []
		cluster_label = []
		nodes = [] #for sorted nodes
		for i,item in enumerate(load_nodes):
			data_size.append(load_nodes[i].data_size)
		sorted_by_data_size = sorted(data_size,reverse=True)
		print('sorted_by_data_size: ', sorted_by_data_size)
		#appending the nodes by data size
		i = 0
		while(i != len(sorted_by_data_size)):
			j = 0
			while(True):
				if(sorted_by_data_size[i] == load_nodes[j].data_size):
					#appending
					nodes.append(load_nodes[j])
					regressor.append(load_regressor[j])
					cluster_label.append(file_name_label[j])
					#deleting
					load_regressor.pop(j)
					load_nodes.pop(j)
					file_name_label.pop(j)
					print('file_name_label: ', file_name_label)
					i += 1
					break
				j += 1
		return nodes,regressor,cluster_label

	def sort_by_reg_error(self,load_nodes,load_regressor,file_name_label):
		'''
		It will sort regressor and node objects based on maximum regression  error  associated with it
		'''
		regressor = []
		reg_error = []
		cluster_label = []
		nodes = [] #for sorted nodes
		for i,item in enumerate(load_nodes):
			reg_error.append(load_nodes[i].max_relerr_train)
		sorted_by_reg_error = sorted(reg_error,reverse=False)
		print('reg_error: ', reg_error)
		print('sorted_by_reg_error: ', sorted_by_reg_error)
		#appending the nodes by data size
		i = 0
		while(i != len(sorted_by_reg_error)):
			j = 0
			while(True):
				if(sorted_by_reg_error[i] == load_nodes[j].max_relerr_train):
					#appending
					nodes.append(load_nodes[j])
					regressor.append(load_regressor[j])
					cluster_label.append(file_name_label[j])
					#deleting
					load_regressor.pop(j)
					load_nodes.pop(j)
					file_name_label.pop(j)
					i += 1
					break
				j += 1
		print('cluster_label: ', cluster_label)
		return nodes,regressor,cluster_label
		

	def filter_data(self,data_passed):
		#finding the error 
		data = copy.deepcopy(data_passed)
		data['rel_error'] =  np.abs(data['y_act'] - data['y_pred'])/np.abs(data['y_act'])
		#Clusters
		data_in_range = data[data['rel_error'] < self.criteria]
		data_outside_range = data[data['rel_error'] > self.criteria]

		data_in_range = data_in_range.drop(columns=['rel_error'])
		data_outside_range = data_outside_range.drop(columns=['rel_error'])
		return data_in_range,data_outside_range

	def check_data_shape(self,nodes):
		for i,item in enumerate(nodes):
			print('\nshape_ :', nodes[i].data_size)
		


	def last_node_encountered(self,data1,nodes,regressor,file_name_label,label_index):
		'''
		If last node encountered then data has to be saved  accordingly
		'''
		print('end node suspected')
		nodes , regre = self.generate_regressor(nodes,data1,file_name_label[label_index],label_index)
		regressor[label_index] = regre
		
		print('self.final_cluster_count: ', self.final_cluster_count)
		self.final_cluster_count += 1

		#storing for output
		self.optimized_nodes.append(nodes[label_index])
		self.optimized_regressor.append(regressor[label_index])
		self.optimized_file_name_label.append(file_name_label[label_index])

		nodes.pop(label_index) #deleting
		regressor.pop(label_index)
		file_name_label.pop(label_index) #deleting
		print('file_name_label: ', file_name_label)
		#sorting by error
		nodes,regressor,file_name_label = self.sort_by_reg_error(nodes,regressor,file_name_label)
		print('file_name_label: ', file_name_label)
		self.check_data_shape(nodes)

		return nodes, regressor, file_name_label
			

	def reduce_nodes(self, nodes, regressor, file_name_label):
		'''
		This method will optimize the nodes
		'''

		new_node_count = 0

		# '''
		# Adding dataframe for left-out points
		# Points which are left in the nodes will be collected in this dataframe and will be served as last node
		# This data frame will be added to data2 frame
		# '''
		left_out_data = pd.DataFrame([])
		
		i = 0 #intialized index
		while(len(nodes) != 0 ):
			print('file_name_label: ', file_name_label)
			print('len(nodes): ', len(nodes))
			data1 = nodes[i].data
			print('data1: ', data1.shape)
			j = 1
			# time.sleep(1)

			## for last node
			if(len(nodes) == 1):

				nodes , regre = self.generate_regressor(nodes,data1,file_name_label[i],i)
				nodes[i].data = data1
				regressor[i] = regre

				#storing for output
				self.optimized_nodes.append(nodes[i])
				self.optimized_regressor.append(regressor[i])
				self.optimized_file_name_label.append(file_name_label[i])
				
				#updating the node count 
				new_node_count += 1
				print('new_node_count: ', new_node_count)
				# time.sleep(3)
				print('only one cluster left')
				data1 = nodes[i].data
				nodes.pop(i) #deleting
				file_name_label.pop(i) #deleting
				print('file_name_label: ', file_name_label)
				break
			
			
			#main Algorithm
			while(True):
				print('\n\n\n start')
				print('j:',j)
				# time.sleep(3)
				print('len(nodes): ', len(nodes))
				self.check_data_shape(nodes)
				if(len(nodes) == 1):
					break

				# time.sleep(1)
				data2 = nodes[j].data
				print('data2: ', data2.shape)

				#adding leftout data to dataframe in hope that if things match we won't lose information 
				data2 = pd.concat([data2,left_out_data],sort=False)
				
				#removing the unnecessary 
				data_pass = copy.deepcopy(data2)
				data_pass = data_pass.drop(columns=['y_act'])
				data_pass = data_pass.drop(columns=['y_pred'])
				print('data_pass: ', data_pass)

				#predict y
				data_pass['y_pred'] = regressor[i].predict(data_pass)
				data_pass['y_act'] = data2['y_act']
				print('predicted : ', data_pass['y_pred'] , 'actual', data_pass['y_act'])


				data_check = copy.deepcopy(data1)
				data_check = data_check.drop(columns=['y_act'])
				data_check = data_check.drop(columns=['y_pred'])
				check = regressor[i].predict(data_check)
				print('predicted : ', check , 'actual', data1['y_act'])

				minimum_rel = data_pass['y_pred'].min()
				print('minimum_rel: ', minimum_rel)
				maximum_rel = data_pass['y_pred'].max()
				print('maximum_rel: ', maximum_rel)
				data_in_range, data_outside_range = self.filter_data(data_pass)


				print('data_outside_range: ', data_outside_range.shape)
				print('data_in_range: ', data_in_range.shape)
				# time.sleep(2)
				#if criteria satisfied appending to frame data-i and rest assigned to data-j
				print('data1: ', data1.shape)
				data1 = pd.concat([data1,data_in_range],sort=False)
				print('data1: ', data1.shape)


				if(data_outside_range.shape[0] < data_outside_range.shape[1]):
					# '''
					# if few data points are left which can't generate model
					# Dump those data and delete the node
					# '''
					#extra data added to left out dataframe
					left_out_data = copy.deepcopy(data_outside_range)  #data added to leftout dataframe
					print('left_out_data: ', left_out_data.shape)

					nodes.pop(j) #deleting
					file_name_label.pop(j) #deleting
					regressor.pop(j) #deleting
					print('\n\n node deleted')
					# time.sleep(3)
					self.check_data_shape(nodes)

					if(j  >= (len(nodes) -1)): 	#greater than in case last node may get deleted from list and index will become bigger then node size
						# '''
						# While searching last node encountered
						# '''
						nodes, regressor, file_name_label  = self.last_node_encountered(data1,nodes,regressor,file_name_label,i)
						#updating node count
						new_node_count += 1
						print('new_node_count: ', new_node_count)
						break

					continue

				elif(data2.shape[0] != data_outside_range.shape[0]):
					# '''
					# If shape is changed and enough data to generate the model then 
					# update the node parameters as associated data is also changed
					# '''
					print('j:',j)
					print('changing the node ')
					# time.sleep(2)
					nodes , regre = self.generate_regressor(nodes,data_outside_range,file_name_label[j],j)
					regressor[j] = regre
					self.check_data_shape(nodes)
					#### Uncomment to verify 
					# y_act = data['y_act']
					# data = data.drop(columns=['y_act'])
					# data = data.drop(columns=['y_pred'])
					# y_pred = regre.predict(data)
					# print('\n y_pred :',y_pred)
					# print('\n y_act :', y_act )
				
				if(j  >= (len(nodes) -1)): 	#greater than in case last node may get deleted from list and index will become bigger then node size
					# '''
					# While searching last node encountered
					# '''
					nodes, regressor, file_name_label = self.last_node_encountered(data1,nodes,regressor,file_name_label,i)
					#updating node count
					new_node_count += 1
					print('new_node_count: ', new_node_count)
					break

				j += 1
				for k,item in enumerate(nodes):
					print(nodes[k].data.shape)

				print('len(nodes): ', len(nodes))
				print('file_name_label: ', file_name_label)
				print('len(nodes): ', len(nodes))
				print('j: ', j)
				print('hello incremented')
				print('new_node_count: ', new_node_count)
				# time.sleep(3)
				print('optimized_nodes: ',len(self.optimized_nodes))
				# time.sleep(2)
	
		return self.optimized_nodes, self.optimized_regressor, self.optimized_file_name_label, new_node_count
			
		
	def dist_calculate_centroid(self,nodes,file_name):
		'''
		This method will calculate distance from centroid
		'''
		i_val = []
		j_val = []
		dist_val =[]
		
		data = pd.DataFrame([])
		for i,item in enumerate(nodes):
			i_centroid = nodes[i].centroid
			for j in range(i,len(nodes)):
				j_centroid = nodes[j].centroid
				dist = self.ref_point.euclidian_dist(i_centroid,j_centroid)
				print('dist: ', dist)

				#APPENDING
				i_val.append(i)
				j_val.append(j)
				dist_val.append(dist)
		print('dist_val: ', dist_val)
		print('trying data frame ')
		data['from'] = i_val
		print('from done')
		data['to'] = j_val
		print('to done')
		data['distance'] = dist_val
		print('dist done')
		
		print('dataframe done')

		SF.check_directory(str(self.curr_directory)+'/result/centroid_dist/')
		data.to_csv(str(self.curr_directory)+'/result/centroid_dist/'+str(file_name)+'.csv')
		print('data sucks')

			
	def optimize_cluster(self):
		'''
		Method will read the nodes and try to combine the data 
		'''
		#cluster path 
		cluster_path = str(self.curr_directory)+'/object_file/leafs/'
		#finding files and its count in the directory 
		num_of_cluster, file_names, file_name_label = self.find_total_clusters(cluster_path)
		print('file_name_label: ', file_name_label)
		load_nodes = []
		load_regressor = [] #objects for prediction
		#loading and storing node
		for i,item in enumerate(file_name_label):
			regre = joblib.load(str(self.curr_directory)+'/object_file/regressor/regressor_'+str(file_name_label[i])+'.sav')
			node = joblib.load(str(cluster_path)+str(file_names[i]))
			load_nodes.append(node)
			load_regressor.append(regre)
		
		
		#sorted regressor and nodes based on size
		nodes, regressor, file_name_label = self.sort_by_reg_error(load_nodes,load_regressor,file_name_label)

		#old_nodes centroid distance
		self.dist_calculate_centroid(nodes,'after_tree')
		print('out pf dist')


		# '''
		# Initial total clusters
		# '''
		initial_clusters = len(nodes)
		print('initial_clusters: ', initial_clusters)
		

		SF.check_directory(str(self.curr_directory)+'/result/final_cluster_result/')
		SF.check_directory(str(self.curr_directory)+'/object_file/final_centroid/')
		SF.check_directory(str(self.curr_directory)+'/object_file/final_regressor/')

		#combine data 
		# '''
		# # Recursively run till number of clusters won't get change 
		# '''
		cluster_size = []
		
		cluster_size.append(initial_clusters)
		

		# '''	
		# load first node-i and then combine data-i with other 
		# node-j and combine data-ij and do regression if error is less then 
		# combine nodes else leave as it and finally purge the node and files
		# from the directory ...
		# store node-i combined with other node to final_leaves 
		# continue till ./leafs directory get emptied
		# '''
		# index
		m=0
		print('before_loop')
		while(True):
			nodes, regressor, file_name_label , new_node_count = self.reduce_nodes(nodes, regressor, file_name_label) #doing it kind of recursively 
			print('file_name_label: ', file_name_label)
			print('new_node_count: ', new_node_count)
			print('old_node_count: ', cluster_size[m])

			cluster_size.append(new_node_count)
			if(cluster_size[m] == cluster_size[m+1]):
				break
			m += 1
			print('optimized_nodes: ',len(nodes))
			print('first loop done')
		

		#old_nodes centroid distance
		self.dist_calculate_centroid(nodes,'after_optimize')
		
		#writing result
		self.save_object_file(nodes,regressor,file_name_label)
		print('cluster_size: ', cluster_size)


	
if __name__ == "__main__":
	import os
	cwd = os.getcwd()
	print('cwd: ', cwd)
	choice = '-t'
	criteria = 0.05
	combine_cluster = combine_clusters(cwd,criteria,choice)
	combine_cluster.optimize_cluster()