'''
This file is useful to generate final external test result as in
pre-print
'''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os

path = './external_test_result/Ignition_delay_comparison/'
files = os.listdir(path)

final_comparision = pd.DataFrame([])
for i in files:
	path_ = path + i
	final_comparision = pd.concat([final_comparision,pd.read_csv(path_,index_col=False)])

#########################
### whole comparision ###
#########################

'''
plot of external test set result 
'''
rel_error_lt_3 = final_comparision[final_comparision['Relative Error'] <= 0.03].shape[0]
rel_error_btn_3_6 = final_comparision[(final_comparision['Relative Error'] > 0.03) & (final_comparision['Relative Error'] <= 0.06)].shape[0]
rel_error_btn_6_9 = final_comparision[(final_comparision['Relative Error'] > 0.06) & (final_comparision['Relative Error'] <= 0.09)].shape[0]
rel_error_btn_9_12 = final_comparision[(final_comparision['Relative Error'] > 0.09) & (final_comparision['Relative Error'] <= 0.12)].shape[0]
rel_error_btn_12_15 = final_comparision[(final_comparision['Relative Error'] > 0.12) & (final_comparision['Relative Error'] <= 0.15)].shape[0]

# x = ['$<= 10\%$ ', '$ 10\% < x <= 20\%$','$ 20\% < x <= 30\%$','$ 30\% < x <= 40\%$','$ 40\% < x <= 50\%$','$ 50\% < x <= 60\%$','$ 60\% < x <= 70\%$','$ 70\% < x <= 80\%$','$ 80\% < x <= 90\%$','$ 90\% < x <= 100\%$','$ 100\% > x $']
x = ['$ 3$ ', '$ 6$','$ 9$','$ 12$','$ 15$']
y = [rel_error_lt_3,rel_error_btn_3_6,rel_error_btn_6_9,rel_error_btn_9_12, rel_error_btn_12_15]
plt.clf()
print(x,y)
fontsize = 19
plt.rc('text', usetex=True)
plt.bar(x,y)
# plt.rc('text', usetex=True)
# plt.rc('xtick',labelsize=15)
# plt.rc('ytick',labelsize=15)
plt.grid(which='minor', alpha=0.2)
plt.tick_params(axis='both', which='minor', labelsize=fontsize)
plt.title('Count of test-data points having \n relative error less than specified criteria',fontsize=fontsize)
plt.xlabel('Relative Error ( $\le$  \%)',fontsize=fontsize)
plt.ylabel('Count of test-data points',fontsize=fontsize)
plt.xticks(fontsize=fontsize, rotation=0)
plt.yticks(fontsize=fontsize, rotation=0)
plt.tight_layout()
plt.savefig('./external_test_result/error_frequency/error_frequency_all_data_filtered.eps', dpi=600)
# plt.show()
plt.close() 

