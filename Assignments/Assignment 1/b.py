import csv
import numpy as np
import pandas as pd
from numpy.linalg import inv
import sys
import math

def squareError(true_vals, predicted_vals):
    '''
        Compute normalized RMSE
        Args:
            true_vals: numpy array of targets
            predicted_vals: numpy array of predicted values
    '''
    # Subtract minimum value
    diff = true_vals-predicted_vals
    error = np.sum(np.square(diff))
    error = error/len(true_vals)
    return error	

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
rowsTrain = len(train)
rowsTest = len(test)
column = len(train[0])
y = np.ones([(rowsTest+rowsTrain)])
y[0:rowsTrain] = train[:,0]
y[rowsTrain:rowsTrain+rowsTest] = test[:,0]

#print(y)

x = np.ones([(rowsTest+rowsTrain),column-1])
x[0:rowsTrain,:] = train[:,1:column]
x[rowsTrain:rowsTrain+rowsTest,:] = test[:,1:column] 


#Both test and train data are merged now. 

#Implement 10-fold cross validation
#Divide data in 2 parts, 1 for train (9/10) and other for test (1/10)
#for each get the lambda which gives the least error, error follows a U-shaped curve against lambda
#save lambda for each of the 10 folds, get the lambda which gives the least error over these 10 
#Implement cross-ridge for the last time with the selected lambda and get the error 

#bestLambda = np.ones([10])
#bestError = np.ones([10])
prevLambda = 0

bestError = math.inf
error=0
lamb = 0
while(True):
	
	error=0
	
	for i in range (0,10):
		k1 = int((9-i)*(rowsTrain+rowsTest)/10)
		k2 = int((10-i)*(rowsTrain+rowsTest)/10)

		xTrain = x[0:k1,:]
		xTrain = np.append(xTrain,x[k2:(rowsTrain+rowsTest),:],axis=0)

		xTest = x[k1:k2,:]

		yTrain = y[0:k1]
		yTrain = np.append(yTrain,y[k2:(rowsTrain+rowsTest)],axis=0)

		yTest = y[k1:k2]
		#add a column of ones in xTrain and in xTest
		rows=len(xTrain)
		columns=len(xTrain[0])
		ones = np.ones([rows,columns+1])
		ones[:,1:columns+1] = xTrain
		xTrain = ones

		rowsTest=len(xTest)
		columnsTest=len(xTest[0])
		onesTest = np.ones([rowsTest,columnsTest+1])
		onesTest[:,1:columnsTest+1] = xTest
		xTest = onesTest
		
		
		v=xTrain.transpose()
		tempX=v.dot(xTrain)
		n = len(xTrain[0])
		identity = np.identity(n)
		
		tempX = tempX + lamb*identity
		tempX = inv(tempX)
		tempX = tempX.dot(v)
		theta = tempX.dot(yTrain)
		yPred = xTest.dot(theta)
		error = error + squareError(yTest,yPred)

	#print(error)	
	if(error<bestError):
		finalLamb = lamb
		bestError = error
		lamb = lamb + 0.0005

	else:
		break

print(finalLamb)

rowTest = (len(test))
columnTest = (len(test[0]))

rowTrain = len(train)
columnTrain = len(train[0])

testX = test[:,1:len(test[0])]
pTest = np.ones([rowTest,columnTest])
pTest[:,1:columnTest] = testX;
testX=pTest
testY = test[:,0]
trainY = train[:,0]


trainX = train[:,1:len(train[0])]
pTrain = np.ones([rowTrain,columnTrain])
pTrain[:,1:columnTrain] = trainX;
trainX=pTrain

v=trainX.transpose()
tempX=v.dot(trainX)
n = len(trainX[0])
identity = np.identity(n)
tempX = tempX + finalLamb*identity
tempX = inv(tempX)
tempX = tempX.dot(v)
theta = tempX.dot(trainY)
predY = testX.dot(theta)

#print(predY)

np.savetxt('outb.txt', predY, fmt="%f")