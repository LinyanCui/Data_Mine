import os
import numpy as np
import pickle
import math
import sys
from sklearn import preprocessing
import time


def sigmoid(inx):
	prob=1.0/(1+math.exp(-inx))
	if prob>0.5:
		return 1.0
	else:
		return 0.0

start = time.clock()


train_x=pickle.load(open("/home/lycui/Adult_Project/train.txt",'r'))
test_x=pickle.load(open("/home/lycui/Adult_Project/test.txt",'r'))
train_x_norm=preprocessing.scale(1.0*train_x[0:13,:],axis=1)
train_x[0:13,:]=train_x_norm
test_x_norm=preprocessing.scale(1.0*test_x[0:13,:],axis=1)
test_x[0:13,:]=test_x_norm

numFeatures,numSamples = np.shape(train_x)
weights=np.ones((numFeatures-1,1),dtype=int)
alpha = 0.01

batchsize=100
epoch=30

count=0
predict=[]
target=[]
per_count=[]

for m in range(epoch):
	iteration=0
	for j in range(numSamples-62):
		fea_now=train_x[:,j]
		fea_now[2]=0
		T=np.dot(fea_now[:-1],weights)
		if T<-500:
			T=-500
		elif T>500:
			T=500

		output=sigmoid(T)

		#print output
		predict.append(output)
		target.append(fea_now[-1])
		#print fea_now[-1] 
		count=count+1
		if count>=batchsize:
			zero_count=0

			predict=np.array(predict,dtype=np.int32)
			target=np.array(target,dtype=np.int32)
			error=predict-target
			#print 'error1:',error
			number=np.shape(error)[0]
			for i in range(number):
				if error[i]==0:
					zero_count=zero_count+1			
			accuracy=zero_count*1.0/number
			iteration=iteration+1
			print 'iteration,epoch:',iteration,m
			print 'accuracy',accuracy
			#print type(error)
			#persion=persion(error[:])
			
			fea=train_x[:,(j-count+1):j+1]
			fea_fin=fea[0:14,:]
			
			error.shape=(1,number)
			error=np.transpose(error)    
		    
			gradient=np.dot(fea_fin,error)
			weights=weights-alpha*gradient

			predict=[]
			target=[]
			count=0

zero_count=0
one_count=0
per_count=[]

Features,Samples = np.shape(test_x)
for j in range(Samples):
	fea_now=test_x[:,j]
	fea_now[2]=0
	T=np.dot(fea_now[:-1],weights)
	if T<-500 :
		T=-500
	elif T>1024:
		T=500

	output=sigmoid(T)
	predict.append(output)
	target.append(fea_now[-1])

predict=np.array(predict,dtype=np.int32)
target=np.array(target,dtype=np.int32)
p_num1=np.count_nonzero(predict==1)
error=predict-target
number=np.shape(error)[0]
print 'number:',number
for i in range(number):
	if error[i]==0:
		zero_count=zero_count+1
		per_count.append(i)			
accuracy=zero_count*1.0/number

p_num1=np.count_nonzero(predict==1) 
p_num2=np.count_nonzero(target==1) 
print 'p_num1:',p_num1

for i in range(number):
	if target[i]==1:
		if predict[i]==1:
			one_count=one_count+1		
print 'one_count:',one_count
persion=one_count*1.0/p_num1
Recall=one_count*1.0/p_num2

#print 'iteratio,epoch:',iteration,epoch
print 'test_accuracy',accuracy
print 'test_persion',persion
print 'test_recall',Recall
print 'test_F1', 2*persion*Recall/(persion +Recall)

elapedtime=time.clock()-start
print 'elapedtime:',elapedtime