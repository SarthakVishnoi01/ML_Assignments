import csv
import numpy as np
import pandas as pd
import sys
from scipy.sparse import csr_matrix

#f= open("outa.txt","w+")
args = sys.argv

#file = r'msd_train.csv'
vocab = pd.read_csv(args[1],header=None)
print(vocab[0][1])

#file2 = r'msd_test.csv'
train = pd.read_csv(args[2],header=None)
print(len(train))
print(train[1][1])
counts = dict()
 #-----------------[column][row]--------------
for i in range(0,len(train)):
	text = str (train[1][i])

	words = text.split()
	for word in words:
	    if word in counts:
	        counts[word] += 1
	    else:
	        counts[word] = 1
print(counts["the"])

#Getting just the values
#train = train.values
#print(train)
#test = test.values
#print(test)

#Getting the y matrix
#y=train[:,0]
#print(y)


#row = (len(train))
#column = (len(train[0]))
#print(row)
#print(column)

#x=train[:,1:len(train[0])]
#print(len(x))
#print(len(x[0]))

#p=np.ones([row,column])
#print(p)


#p[:,1:91] = x
#print(p)

#x=p
#print(x)

#v=x.transpose()

#from numpy.linalg import inv
#b=v.dot(x)
#c=inv(b)
#d=c.dot(v)

#theta=d.dot(y)
#print(theta)

#predicting with given values on test data
#rowTest = (len(test))
#columnTest = (len(test[0]))

#xTest = test[:,1:len(test[0])]
#pTest = np.ones([rowTest,columnTest])
#pTest[:,1:91] = xTest;
#xTest=pTest
#yPred = xTest.dot(theta)

#np.savetxt(args[3], yPred, fmt="%f")