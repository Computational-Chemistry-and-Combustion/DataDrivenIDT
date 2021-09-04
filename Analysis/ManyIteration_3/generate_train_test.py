import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os 

# ##########################################
# #### for c9 and  C16 scheme
# data = pd.read_csv('dataset.csv')
# train,test = train_test_split(data,test_size=0.1)
# train.to_csv('train.csv',index=False)


# other_fuel = pd.read_csv('other_fuel.csv')
# test = pd.concat([test,other_fuel])
# test.to_csv('test.csv',index=False)


## files in dir
files = os.listdir('./Clusters')

train = pd.DataFrame([])
test = pd.DataFrame([])


for i in files:
	print(i)
	df = pd.read_csv('./Clusters/'+i)
	train_,test_ = train_test_split(df,test_size=0.05)
	train = pd.concat([train_,train],sort=False)
	test = pd.concat([test,test_],sort=False)




## for normal s
train.to_csv('trainset.csv',index=False)
test.to_csv('testset.csv',index=False)