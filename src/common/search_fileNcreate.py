#tracking Current directory 
import os
current_directory  = os.getcwd()
# print('current_directory: ', current_directory)

#Addding all files from src to system default directory 
import sys
sys.path.insert(0, str(current_directory)+'/src')

import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
# print('dir_path: ', dir_path)
sys.path.append(dir_path)
sys.path.append(dir_path+str('/clustering_methods'))


#Obtaining Path of directory 
dir_split = dir_path.split('/')
# print('dir_split: ', dir_split)
Main_folder_dir = ''
for i in range(len(dir_split)-1):
    Main_folder_dir += dir_split[i] + str('/')

class search_fileNcreate():

    def check_file_existence(file_name=None):
        # print('file_name: ', file_name)
        file_dir = str(file_name)
        try:
            os.mknod(file_dir)
            # print('\n File does not exist so, created!\n')
        except FileExistsError:
            # print('\n File already exist!')
            pass
    
    def check_directory(folder_name=None):
        try:
            os.makedirs(folder_name)
            # print('\n Directory does not exist so, created!\n')
        except FileExistsError:
            # print('\n Directory already exist!\n ')     
            pass     

    def check_directory_N_AddNum(folder_name=None):
        '''
        If directory already exist then it chnage name of directory by adding number
        ./result
        ./result_1
        ./result_2 
        '''
        def nextnonexistent(f):
            fnew = f
            root, ext = os.path.splitext(f)
            i = 0
            while os.path.exists(fnew):
                i += 1
                fnew = '%s_%i%s' % (root, i, ext)
            return fnew

        dir_name = nextnonexistent(folder_name)
        try:
                os.makedirs(dir_name)
        except FileExistError:
                pass
        return dir_name
        
if __name__ == "__main__":
    search_fileNcreate.check_file_existence("result/bond_detail.csv")
    search_fileNcreate.check_directory_N_AddNum("result/app/dir")
