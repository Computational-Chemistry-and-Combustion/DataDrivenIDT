'''
This is a main method for all .py files
this files control all other files and controlled 
by shell script
'''

#tracking Current directory 
import os
import sys
from Flag import Flag
from search_fileNcreate import search_fileNcreate as SF

current_directory  = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
sys.path.append(dir_path+str('/clustering_methods'))
sys.path.insert(0, str(current_directory)+'/src') #Addding all files from src to system default directory 



#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
    Main_folder_dir += dir_split[i] + str('/')


########################Change the data here #########################
arguments_len = len(sys.argv)
# print('arguments_len: ', arguments_len)
flag = sys.argv[1]
method = sys.argv[2]
process = sys.argv[3]

if(flag == '-b'): #only -b flag has fuel, other has dataset location
    smile = sys.argv[4]
else:
    dataset_location = sys.argv[4]
curr_directory = sys.argv[5]

# print('curr_directory: ', curr_directory)

#cleaning output file
SF.check_directory(str(curr_directory)+"/result" ) #checking directory
SF.check_file_existence(str(curr_directory)+"/result/output_result.txt") #checking file existence
f = open(str(curr_directory)+"/result/output_result.txt", "w")
f.write('')
f.close()

# print('control in DDS')

if(method == 'common'):
    if(flag != '-b'):
        if(flag == '-a'):
            analysis_type = sys.argv[6]
            Flag.switch_func(flag,method,process,dataset_location=dataset_location,curr_directory=curr_directory,analysis_type=analysis_type)   
        else:
            Flag.switch_func(flag,method,process,dataset_location=dataset_location,curr_directory=curr_directory) 
    elif(flag == '-b'):
        Flag.switch_func(flag,curr_directory=curr_directory,smile=smile) 
    else:
        print('PASS CORRECT FLAGS')
        exit()

if(method == 'tree'):
    if(process == 'train'):
        #arguments passed by Run.sh
        #error based tree criteria
        error_criteria_to_divide = float(sys.argv[6])
        #elimination criteria 
        elimination = sys.argv[7]
        elimination = True if elimination=='True' else False
        #backward elimination significance criteria 
        sl = float(sys.argv[8])
        #If you want to use significance level
        limited_ref_points = sys.argv[9]
        if(limited_ref_points == 'True' or limited_ref_points == 'False'):
            limited_ref_points = True if limited_ref_points=='True' else False
        Flag.switch_func(flag,method,process,dataset_location=dataset_location,curr_directory=curr_directory,division_error_criteria=error_criteria_to_divide,elimination=elimination,sl=sl,limited_ref_points=limited_ref_points) 
    if(process == 'test'):
        Flag.switch_func(flag,method,process,dataset_location=dataset_location,curr_directory=curr_directory) 



if(method == 'GMM'):
    if(process == 'train'):
        #arguments passed by Run.sh
        n_cluster = sys.argv[6]
        max_iter = int(sys.argv[7])
        limited_ref_points = sys.argv[8]
        if(limited_ref_points == 'True' or limited_ref_points == 'False'):
            limited_ref_points = True if limited_ref_points=='True' else False
        Flag.switch_func(flag,method,process,dataset_location=dataset_location,curr_directory=curr_directory,n_cluster=n_cluster,limited_ref_points=limited_ref_points)
    if(process == 'test'):
        Flag.switch_func(flag,method,process,dataset_location=dataset_location,curr_directory=curr_directory) 
        
if(method == 'spath'):
    if(process == 'train'):
        #arguments passed by Run.sh
        n_cluster = sys.argv[6]
        max_iter = int(sys.argv[7])
        limited_ref_points = sys.argv[8]
        if(limited_ref_points == 'True' or limited_ref_points == 'False'):
            limited_ref_points = True if limited_ref_points=='True' else False
        Flag.switch_func(flag,method,process,dataset_location=dataset_location,curr_directory=curr_directory,n_cluster=n_cluster,max_iter=100,limited_ref_points=limited_ref_points)
    if(process == 'test'):
        Flag.switch_func(flag,method,process,dataset_location=dataset_location,curr_directory=curr_directory) 

if(method == 'multi'):
    if(process == 'train'):
        #arguments passed by Run.sh
        print('Running multiple regression algorithm:')
        elimination=sys.argv[6]
        elimination = True if elimination=='True' else False
        sl = float(sys.argv[7])
        Flag.switch_func(flag,method,process,dataset_location=dataset_location,curr_directory=curr_directory,elimination=elimination,sl=sl)
    if(process == 'test'):
        Flag.switch_func(flag,method,process,dataset_location=dataset_location,curr_directory=curr_directory) 