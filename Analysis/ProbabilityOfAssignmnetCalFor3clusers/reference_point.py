import numpy as np
import matplotlib.pyplot as plt
import subprocess
import time
import copy
from search_fileNcreate import search_fileNcreate as SF
import joblib


class reference_point():
	'''
	This method will find out the reference points from
	like
	centroid,
	extreme points,
	other points based on euclidean distance 
	'''
	def __init__(self,curr_directory,criteria,choice,limited_ref_points = False,refPoints=10):
		self.choice = choice
		self.curr_directory = curr_directory
		self.criteria = criteria
		self.limited_ref_points = limited_ref_points

 
	def calculate_centroid(self,data):
		'''
		It will calcute the centroid using all the data points 
		'''
		#removing y_act , y_pred and constant column to calculate the centroid
		data_centroid = copy.deepcopy(data) #coping the data
		data_centroid = data_centroid.drop(columns=['y_act'])
		data_centroid = data_centroid.drop(columns=['y_pred'])
		data_centroid = data_centroid.drop(columns=['Constant'])
		try:
			if(data_centroid.empty is True):
				return None
			# print('calculating centroid')
			num_data_points = data_centroid.shape[0]
			centroid = data_centroid.sum(axis=0) / num_data_points  #columnwise sum or along rows sum
			# print('centroid: ', centroid)
			return centroid
		except AttributeError:
			if(data_centroid is None):
				return None

	def other_reference_point(self,data,centroid,child_label,dir_name='final_cluster_reference_points'):
		'''
		It will calculate the other reference point apart from centroid.
		1. farthest point from centroid
		2. nearest point from centroid
		3. farthest point from origin
		4. nearest point from origin
		5. Other extreme points by mentod find_extreme_point
		for specific cluster or data passed.
		return : all four specified points

		dir_name = if default --- generate for optimized nodes if passed then can work
		'''
		if(centroid.all() is not None):
			data_passed = copy.deepcopy(data) #just copy of data points
			data_passed = data_passed.drop(columns=['y_act'])
			data_passed = data_passed.drop(columns=['y_pred'])
			data_passed = data_passed.drop(columns=['Constant'])
			#resetting index
			data_passed = data_passed.reset_index(drop=True)

			distance_from_centroid = []
			distance_from_origin = []
			origin = np.zeros(centroid.shape[0]) #origin point defined
			for i in range(len(data_passed)):
				#distance from centroid
				distance_from_centroid.append(self.euclidian_dist(data_passed.loc[i,:],centroid))
				#distance from origin
				distance_from_origin.append(self.euclidian_dist(data_passed.loc[i,:],origin))

			#Distance from centroid to point & origin to point----findind index of data 
			max_distance_from_centroid_PointIndex = np.argmax(distance_from_centroid)
			min_distance_from_centroid_PointIndex  = np.argmin(distance_from_centroid)
			max_distance_from_origin_PointIndex    = np.argmax(distance_from_origin)
			min_distance_from_origin_PointIndex    = np.argmin(distance_from_origin)


			#based on index finding the point
			Point_max_dist_from_centroid = data_passed.loc[max_distance_from_centroid_PointIndex,:]
			Point_min_dist_from_centroid = data_passed.loc[min_distance_from_centroid_PointIndex,:]
			Point_max_dist_from_origin = data_passed.loc[max_distance_from_origin_PointIndex,:]
			Point_min_dist_from_origin = data_passed.loc[min_distance_from_origin_PointIndex,:]


			SF.check_directory(str(self.curr_directory)+'/object_file/'+str(dir_name)+'/')
			######  Storing above points as object files  ##############
			#filenames naming convention just for sake of ease
			filename_max_dist_centroid =  str(self.curr_directory)+'/object_file/'+str(dir_name)+'/maxCentroid_'+str(child_label)+'.sav'
			filename_min_dist_centroid =  str(self.curr_directory)+'/object_file/'+str(dir_name)+'/minCentroid_'+str(child_label)+'.sav'
			filename_max_dist_origin  =  str(self.curr_directory)+'/object_file/'+str(dir_name)+'/maxOrigin_'+str(child_label)+'.sav' #
			filename_min_dist_origin  =  str(self.curr_directory)+'/object_file/'+str(dir_name)+'/minOrigin_'+str(child_label)+'.sav'
			#dumping object files
			# joblib.dump(Point_max_dist_from_centroid,filename_max_dist_centroid )
			# joblib.dump(Point_min_dist_from_centroid,filename_min_dist_centroid ) 
			# joblib.dump(Point_max_dist_from_origin,filename_max_dist_origin ) 
			# joblib.dump(Point_min_dist_from_origin,filename_min_dist_origin ) 

			#other reference point
			print('Finding extreme points')
			time.sleep(5)
			other_ref_points = self.find_extreme_point(data_passed,Point_max_dist_from_centroid,centroid)
			for i,item in enumerate(other_ref_points):
				filename_max_dist_centroid =  str(self.curr_directory)+'/object_file/'+str(dir_name)+'/other_refPoi_'+str(i)+'_'+str(child_label)+'.sav'
				joblib.dump(other_ref_points[i],filename_max_dist_centroid )
		else:
			pass
	    
	def find_extreme_point(self,data_passed,max_dist_point,centroid):
		'''
		From supplied points and data it will calculate the farthest point.
		let's say I want 10 such point and starting point is farthest point-A from 
		centroid of data and now I want to find out 10 extreme points.
		so, 
		1. Calculate the point-B farthest from point-A
		2. Calculate point-c fathest from point-B and point-A
		3. repeat the procedure 
		return all the points
		'''
		data =  copy.deepcopy(data_passed)
		ref_points = []
		#take reference point as initial_point
		ref_points.append(max_dist_point)
		ref_points.append(centroid)
		print('start')
		## After appending the point purge it
		data = data.drop(max_dist_point.name)
		####how many times run the loop
		#try with dimension^2 but if less data then consider all the data points
		if(self.limited_ref_points is not False and self.limited_ref_points is not True): ##it will use limited point of fuel as reference point
				print('All the points are used as reference points')
				if(data.shape[0] >= data.shape[1] * int(self.limited_ref_points)): 
					num_ref_point = data.shape[1] * int(self.limited_ref_points)
					for i in range(num_ref_point):
						measured_dist_from_all_ref_point = []
						for j in range (data.shape[0]): #for all the data points 
							dist_sum = 0
							data = data.reset_index(drop=True)
							for k,item in enumerate(ref_points): #distance from all the points
								dist_sum += self.euclidian_dist(ref_points[k],data.loc[j,:])
							#for data point-j distance is measured from all the distance and ref_points
							measured_dist_from_all_ref_point.append(dist_sum)
						#after storing all measures dist. find index of list and locate that data point
						Point_index = np.argmax(measured_dist_from_all_ref_point)
						#find data point based on index
						data_point = data.loc[Point_index,:]
						data_point = data_point.reset_index(drop=True)
						#dropping point from the data 
						data = data.drop(Point_index)
						#append that data point as reference point
						ref_points.append(data_point)
				return ref_points

		elif(data.shape[1] == 1):  #for 1D data
			# time.sleep(10)    		
			print('Working with 1d data')
			####two reference point one extreme and centroid is already added  
			num_ref_point = 2
			for i in range(num_ref_point):
				measured_dist_from_all_ref_point = []
				for j in range (data.shape[0]): #for all the data points 
					# print('\n\n\nlen(ref_points): ', len(ref_points))
					# print('ref_points: ', ref_points)
					dist_sum = 0
					data = data.reset_index(drop=True)
					for k,item in enumerate(ref_points): #distance from all the points
						# print('data.loc[j]: ', data.loc[j])
						# print('ref_points[k]: ', ref_points[k])
						dist_sum += self.euclidian_dist(ref_points[k],data.loc[j])
						# print('dist_sum: ', dist_sum)
						# time.sleep(2)
					# print('total_dist :' , dist_sum)
					#for data point-j distance is measured from all the distance and ref_points
					# print('ref_points: ', ref_points)
					measured_dist_from_all_ref_point.append(dist_sum)
					# print('measured_dist_from_all_ref_point: ', measured_dist_from_all_ref_point)
			#after storing all measures dist. find index of list and locate that data point
			Point_index = np.argmax(measured_dist_from_all_ref_point)
			# print('Point_index: ', Point_index)
			#find data point based on index
			data_point = data.loc[Point_index]
			# print('data_point: ', data_point)
			data_point = data_point.reset_index(drop=True)
			#dropping point from the data 
			data = data.drop(Point_index)
			# print('data: ', data)
			# time.sleep(10)
			#append that data point as reference point
			ref_points.append(data_point)
			return ref_points
			
		elif(self.limited_ref_points is True):		
			print('All the points are used as reference points')
			num_ref_point = data.shape[0]
			data = data.reset_index(drop=True)
			for i in range(data.shape[0]):
			# for i in range(5):
				data_point = data.loc[i,:]
				data_point = data_point
				ref_points.append(data_point)
			return ref_points

		elif(self.limited_ref_points is False):
			return ref_points
		    
	def euclidian_dist(self,arr_1,arr_2):
		arr_1 = np.array(arr_1)
		arr_2 = np.array(arr_2)

		# '''
		# calculating distance by passed row of matrix and centroid 
		# '''
		distance = np.linalg.norm(arr_1-arr_2)
		return distance