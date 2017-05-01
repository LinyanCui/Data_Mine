import os
import pickle
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import numpy as np



trainFile='train.data'
testFile='test.data'
trainpath="/home/lycui/Adult_Project/test.txt"
names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]

train_data=pd.read_csv('/home/lycui/Adult_Project/test.data',names=names)
data=train_data[names]
train_info=[]
trainTxt=open(trainpath,'a+')
#print data.age
#print data['income']
for column in names:
	m=LabelEncoder()
	m.fit(data[column])
	print data[column]
	print m.transform(data[column])
	train_info.append(m.transform(data[column]))
print train_info
#print train_info[14][2]
train_info=np.array(train_info)
#print train_info[0]

pickle.dump(train_info,open(trainpath,'a+'))
