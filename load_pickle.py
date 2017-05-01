import pickle
import numpy as np

obj=pickle.load(open("/home/lycui/Adult_Project/train.txt",'r'))
print np.shape(obj)