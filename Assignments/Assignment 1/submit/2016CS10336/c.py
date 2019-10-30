import csv
import numpy as np
import pandas as pd
import sys
from sklearn import linear_model

#f= open("outa.txt","w+")

args = sys.argv

#file = r'args[1]'
train = pd.read_csv(args[1],header=None)
file2 = r'msd_test.csv'
test = pd.read_csv(args[2],header=None)
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
reg = linear_model.LassoLars(alpha=0.000001, eps=2.220446049250313e-20, max_iter=100)
reg.fit(x,y)

#Prediction on new data
rowTest = (len(test))
columnTest = (len(test[0]))

xTest = test[:,1:len(test[0])]
pTest = np.ones([rowTest,columnTest])
pTest[:,1:91] = xTest;
xTest=pTest
yPred = reg.predict(xTest)

np.savetxt(args[3], yPred, fmt="%f")
