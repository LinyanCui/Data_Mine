import os
import os
import pandas as pd 
import pickle  
import seaborn as sns     # Statistical visualization library based on Matplotlib
import matplotlib.pyplot as plt   # MATLAB-like plotting, useful for interactive viz
import json


#trainFile='/home/lycui/Adult_Project/adult.data'
#trainFile="/home/lycui/Adult_Project/try"
testFile='/home/lycui/Adult_Project/adult.test'
#train_txtPath="/home/lycui/Adult_Project/train.data"
test_txtPath="/home/lycui/Adult_Project/test.data"
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

#traindata=open(testFile)
testdata=open(testFile)
Tran_info=[]
train_txt=open(test_txtPath,'a+')
count=0
while 1:   
    #line=traindata.readline()
    line=testdata.readline() 
    if not line:
        break
    line_old=line.strip('.\r\n').split(',')
    #print line
    if ' ?' in line_old:
        print 'find'
        continue
 
    train_txt.write(line)
    count=count+1
    print count
    #Tran_info.append(line)

    #print Tran_inro[0][0]
    
