import csv
import numpy as np
import pandas as pd
import sys
from sklearn import linear_model

#f= open("outa.txt","w+")

file = r'msd_train.csv'
train = pd.read_csv(file,header=None)
file2 = r'msd_test.csv'
test = pd.read_csv(file2,header=None)
#print(df)

#Getting just the values
train = train.values
#print(train)
test = test.values
#print(test)

#Getting the y matrix
y=train[:,0]
#print(y)


row = (len(train))
column = (len(train[0]))
#print(row)
#print(column)

x=train[:,1:len(train[0])]
#print(len(x))
#print(len(x[0]))

p=np.ones([row,column])
#print(p)


p[:,1:91] = x
#print(p)

x=p
#print(x)
for i in range(10,60):
	for j in range (10,60):
		#xTemp = np.ones([len(x),1])
		xTemp = x[:,i]*x[:,j]
		#print(xTemp)
		np.c_[x,xTemp]

reg = linear_model.LassoLars(alpha=0.00001, eps=2.220446049250313e-20, max_iter=500)
reg.fit(x,y)

#Prediction on new data
rowTest = (len(test))
columnTest = (len(test[0]))

xTest = test[:,1:len(test[0])]
pTest = np.ones([rowTest,columnTest])
pTest[:,1:91] = xTest;
xTest=pTest
for i in range(10,60):
	for j in range (10,60):
		xTemp = xTest[:,i]*xTest[:,j]
		np.c_[xTest,xTemp]

yPred = reg.predict(xTest)

np.savetxt('outc.txt', yPred, fmt="%f")
