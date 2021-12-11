##Directory to export the file of combination of different files

###############
# file_name   #
###############
file_name = 'MaxRelError.tex'



dir_path = './../'

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
# print('dir_path: ', dir_path)
sys.path.append(dir_path)


#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
        Main_folder_dir += dir_split[i] + str('/')

Main_folder_dir = Main_folder_dir + str('plots/')


########################Chnage the data here #########################
print(sys.argv)
arguments_len = len(sys.argv)
print('arguments_len: ', arguments_len)
curr_directory = sys.argv[1]
print('file_name: ', curr_directory)
######################################################################


#opening file to read READING MODE
file = open(curr_directory+file_name, 'r')
file_data = file.read()
file.close()    #closing as it's read mode only
#opening file to read WRITING MODE to write changes
file = open(Main_folder_dir+file_name, 'w')
file_data = file_data.replace("{['", "{")
file_data = file_data.replace("']}", "}")
file_data = file_data.replace("'\n '", ",\\\\ '")
file_data = file_data.replace("' '", ",\\\\ '")
file_data = file_data.replace("'", " ")
file.write(file_data)
file.close()
