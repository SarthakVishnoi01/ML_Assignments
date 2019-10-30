import csv
import numpy as np
import scipy as sc
import pandas as pd
import math
import sys
from numpy.linalg import inv
import time
import keras
#def train(X,Y,batch_size,eta,max_iterations):
#CHANGE HOW YOU TAKE K< DO ONE PASS OVER THE ENTIRE Y VECTOR
def sigmoid (x):
	x = np.array(x)
	return 1.0/(1.0+np.exp(-x))
def relu (x):
	x = np.array(x)
	return x * (x > 0)
def tanh(x):
	x = np.array(x)
	return np.tanh(x)
def activation(X,name):
	if(name=="relu"):
		return relu(X)
	elif(name=="sigmoid"):
		return sigmoid(X)
	else:
		return tanh(X)

#Here x will be the value after activation
def sigmoidGradient(x):
	x = np.array(x)
	return x*(1-x)
def reluGradient(x):
	x=np.array(x)
	if(x.all()==0):
		return 0
	else: 
		return 1
def tanhGradient(x):
	x = np.array(x)
	return  (1-x*x)
def gradient(X,name):
	if(name=="relu"):
		return reluGradient(X)
	elif(name=="sigmoid"):
		return sigmoidGradient(X)
	else:
		return tanhGradient(X)

#f= open("outa.txt","w+")
startTheCode = time.time()
args = sys.argv

#file_read = r'devnagri_train.csv'
#Reading the training data and test data

train = pd.read_csv(args[1],header=None)

#file2 = r'devnagri_test_public.csv'
test = pd.read_csv(args[2],header=None)	
#test = pd.read_csv(file_read_test,header=None)
test = test.values
#file2 = r'msd_test.csv'
#test = pd.read_csv(args[2],header=None)

#Getting just the values
train = train.values
#print(train)
#test = test.values
#print(test)

#Getting the y matrix
np.random.shuffle(train)
y = train[:,0]
k=0
for i in range(len(y)):
	if(y[i]>k):
		k = y[i]
k = k+1
#print("This is k", k)
#print("n=",len(y))
x = train[:,1:len(train[0])]
n = len(x); #Number of Examples
m = len(x[0]); #Number of Features
#np.savetxt("x.txt", x, fmt="%d")
#Taking input is completed from file
p = np.zeros([n,k])
for i in range(n):
	o = y[i];
	p[i][o] = 1
y = p
#np.savetxt("y.txt", y, fmt="%d")
#print(len(y[0]))
#Normalise the input data

#SHUFFLE THE TRAINING DATA
x = x/255;

lambdaa = 0.01 #Regularisation
#m=1024
batchSize = int(args[4]) #100 #Take from command line
etaKnot = float(args[5]) #0.01 #take from command line
activationFunction = args[6] #"sigmoid" #Take from command line
hLayers = args[7:] #Taken from coomand line
hiddenLayers = list(map(int,hLayers))
hiddenLayersSize = len(hiddenLayers)
layerSizes = [m]
for i in range(hiddenLayersSize):
	layerSizes.append(hiddenLayers[i])
layerSizes.append(k)
#layerSizes = layerSizes+1
#print("LayerSizes", layerSizes)
#print(hiddenLayerSize)
#print([1,2])
#print(x[0])
#print("Reading of Files Done!")
#Initialising weight matrix:- Number of weight matrices = hiddenLayerSize+1
#maxxDimension = maxDimension(hiddenLayers, k, m)

weights = np.array([np.array([np.array([(1.0*np.random.random_sample()-0.5) for p in range(layerSizes[i]+1)]) for j in range(layerSizes[i+1])]) for i in range(len(layerSizes)-1)]) 
#print(len(weights))
#x=[[1,2],[1,2,3],[1]]
#y=numpy.array([numpy.array(xi) for xi in x])
#print(len(weights))
#np.savetxt("initWeights.txt", weights[1], fmt="%f")
#layersBeforeActivation = np.array([np.array([(1.0*np.random.random_sample()-0.5) for p in range(layerSizes[i]+1)]) for i in range(len(layerSizes))])
layersActivation = np.array([np.array([np.array([(1.0*np.random.random_sample()-0.5) for e in range(batchSize)])for p in range(layerSizes[i]+1)]) for i in range(len(layerSizes))])
errorTerm = np.array([np.array([np.array([(1.0*np.random.random_sample()-0.5) for e in range(batchSize)])for p in range(layerSizes[i]+1)]) for i in range(len(layerSizes))])

gradientOfCost = np.array([np.array([np.array([0.0 for p in range(layerSizes[i]+1)]) for j in range(layerSizes[i+1])]) for i in range(len(layerSizes)-1)]) 
capitalTriangle = np.array([np.array([np.array([0.0 for p in range(layerSizes[i]+1)]) for j in range(layerSizes[i+1])]) for i in range(len(layerSizes)-1)])

inputData = np.array([np.array([0.0 for p in range(batchSize)]) for i in range(m+1)])
outputData = np.array([np.array([0.0 for p in range(batchSize)]) for i in range(k)])

# print(len(inputData))
# print(len(inputData[0]))

start = time.time()
#n = int(5*n/6)
for numberOfIterations in range (22):
	print("Epoch number is", numberOfIterations)
	for i in range(int(n/batchSize)):
		p = i*batchSize
		capitalTriangle = capitalTriangle - capitalTriangle

		inputData[0][:] = np.array([1.0 for e in range(batchSize)])	
		inputData[1:][:] = (x[p:p+batchSize][:]).transpose() #1025*100

		outputData = (y[p:p+batchSize]).transpose() #46*100

		#layersActivation[0][0][:] = np.array([1.0 for e in range(batchSize)])
		layersActivation[0] = inputData
		
		for v in range(1,len(layerSizes)-1):
			layersActivation[v][0][:] = np.array([1.0 for e in range(batchSize)])
			layersActivation[v][1:] = activation(np.dot(weights[v-1],(layersActivation[v-1])),activationFunction)
		v = len(layerSizes)-1
		layersActivation[v][0][:]=np.array([1.0 for e in range(batchSize)])
		layersActivation[v][1:] = activation(np.dot(weights[v-1],(layersActivation[v-1])),"sigmoid")
		
		errorTerm[0] = np.array([np.array([0.0 for p in range(batchSize)]) for i in range(m+1)])
		errorTerm[len(layerSizes)-1][1:] = (layersActivation[len(layerSizes)-1][1:] - outputData)
		
		for v in range(len(layerSizes)-2,0,-1):
			errorTerm[v] = np.dot((weights[v].transpose()),(errorTerm[v+1][1:])) * gradient(layersActivation[v],activationFunction)
			capitalTriangle[v] = capitalTriangle[v] + np.dot(errorTerm[v+1][1:],(layersActivation[v].transpose()))
			#capitalTriangle[v] = capitalTriangle[v] + np.outer(errorTerm[v+1][1:],layersActivation[v])
		capitalTriangle[0] = capitalTriangle[0] + np.dot(errorTerm[1][1:],(layersActivation[0]).transpose())	

		gradientOfCost = (capitalTriangle + lambdaa*weights)/batchSize
		weights = weights - etaKnot * gradientOfCost #See how to change this

finish = time.time()
#print("Time taken is", (finish-start))

# np.savetxt("secondWeights.txt", weights[1], fmt="%f")
#tempLayer = layersActivation[1]	
#np.savetxt("secondActivationLayer.txt", tempLayer, fmt="%f")
#Training done
#args[7]

#Use this data to predict the new inputs

#Test on train data only


#print("Backpropogation Done!")
#Kitno par test karna hai, first testNumber par
####################################START ON TEST DATA##################################################################################
#testNumber = n



yTest = test[:,0]

xTest = test[:,1:len(test[0])]
nTest = len(xTest); #Number of Examples
#print("nTest is", nTest)
mTest = len(xTest[0]); #Number of Features
#print("mTest is", mTest)

# nTest = 1
# xTest = test[0:nTest]
xTest = xTest/255
#print("Layer Sizes is", layerSizes)
pTest = np.zeros([nTest,46])
for i in range(nTest):
	o = yTest[i];
	pTest[i][o] = 1
yTest = pTest
yAnswer = np.zeros([nTest])
#layersBeforeActivationTest = np.array([np.array([(1.0*np.random.random_sample()-0.5) for p in range(layerSizes[i]+1)]) for i in range(len(layerSizes))])
layersActivationTest = np.array([np.array([np.array([0.0 for e in range(nTest)])for p in range(layerSizes[i]+1)]) for i in range(len(layerSizes))])
#print("Before",len(layersActivationTest[0]))
#layersActivation[0][0][:] = np.array([1.0 for e in range(batchSize)])

layersActivationTest[0][1][:] = np.array([1.0 for e in range(nTest)])
layersActivationTest[0][1:][:] = xTest.transpose();

# print("xTest", len(xTest))
# print("xTest[0]", len(xTest[0]))
# print("AfterAssignment Activation Layer", len(layersActivationTest[0]))

for v in range(1,len(layerSizes)-1):
	layersActivationTest[v][0][:] = np.array([1.0 for e in range(nTest)])
	layersActivationTest[v][1:] = activation(np.dot(weights[v-1],(layersActivationTest[v-1])),activationFunction)

v = len(layerSizes)-1
layersActivationTest[v][0][:]=np.array([0.0 for e in range(nTest)])
layersActivationTest[v][1:] = activation(np.dot(weights[v-1],(layersActivationTest[v-1])),"sigmoid")

tempLayerr = np.array([0.0 for e in range(k)])
for ll in range(k):
	tempLayerr[ll] = layersActivationTest[2][ll][0]
#tempLayerr = layersActivationTest[2][:][0]
temptemp = layersActivationTest[2][0][:]
yAnswer = np.array([0 for e in range(nTest)])

for outer in range(nTest):
	maxx= 0.0
	answer=0
	for inner in range(k+1):
		if(layersActivationTest[v][inner][outer] > maxx):
			maxx = layersActivationTest[v][inner][outer]
			answer = inner
	yAnswer[outer] = answer-1
#np.savetxt("tempLayerr.txt", tempLayerr, fmt="%f")
#np.savetxt("temptemp.txt", temptemp, fmt="%f")
#yAnswer = np.amax(layersActivationTest[v], axis=0)


	# if i==0:
	# 	np.savetxt(args[12],layersActivationTest[0][1:] , fmt="%f")
	# 	np.savetxt(args[13],layersActivationTest[1][1:] , fmt="%f")
	# 	np.savetxt(args[14],layersActivationTest[2][1:] , fmt="%f")
		#np.savetxt("fourthActivationLayerTest.txt", layersActivationTest[3][1:], fmt="%f")
		#np.savetxt("fifthActivationLayerTest.txt", layersActivationTest[4][1:], fmt="%f")
	# v = len(layerSizes)-1
	# for j in range (1,k+1):
	# 	if(layersActivationTest[v][j] > maxxx):
	# 		maxxx = layersActivationTest[v][j]
	# 		answer = j-1
	# yAnswer[i] = answer		
	#temp_answer = np.zeros([46])
	#temp_answer[answer] = 1
	#yTest[i] = temp_answer

#weights[len(layerSizes)-2]
np.savetxt(args[3],yAnswer , fmt="%d")
#np.savetxt(args[8], yAnswer, fmt="%d")
# np.savetxt(args[10], weights[0], fmt="%f")
# np.savetxt(args[11], weights[1], fmt="%f")
#np.savetxt(args[12], weights[len(layerSizes)-2], fmt="%f")

#print("Atleast Compile toh hua!!")
finishTheCode = time.time()
#print("Total time taken", finishTheCode-startTheCode)
# b=v.dot(x)
# c=inv(b)
# d=c.dot(v)

# theta=d.dot(y)
# print(theta)

# #predicting with given values on test data
# rowTest = (len(test))
# columnTest = (len(test[0]))

# xTest = test[:,1:len(test[0])]
# pTest = np.ones([rowTest,columnTest])
# pTest[:,1:91] = xTest;
# xTest=pTest
# yPred = xTest.dot(theta)

# np.savetxt(args[3], yPred, fmt="%f")
# for i in range(0,nTest):
# 	#layersBeforeActivationTest[0][0]=1
# 	# mu = sum(x[i])/m;
# 	# x[i] = x[i] - mu
# 	# sigma2 = np.sqrt(np.sum(np.square(x[i])))
# 	# x[i] = x[i]/sigma2
	
# 	layersActivationTest[0][0]=1
# 	#layersBeforeActivationTest[0][1:(m+1)] = x[i][:]
# 	layersActivationTest[0][1:] = xTest[i][:]
# 	for v in range(1,len(layerSizes)):
# 		#layersBeforeActivationTest[v][0] = 1
# 		layersActivationTest[v][0] = 1
# 		#layersBeforeActivationTest[v][1:layerSizes[v]+1] = (weights[v-1].dot(layersAfterActivationTest[v-1]))
# 		layersActivationTest[v][1:] = activation(np.dot(weights[v-1],(layersActivationTest[v-1])),activationFunction)
# 	v = len(layerSizes)-1	
# 	#layersBeforeActivationTest[v][0] = 1
# 	layersActivationTest[v][0] = 1
# 	#layersBeforeActivationTest[v][1:layerSizes[v]+1] = (weights[v-1].dot(layersAfterActivationTest[v-1]))
# 	layersActivationTest[v][1:] = activation(np.dot(weights[v-1],(layersActivationTest[v-1])),"sigmoid")
# 	#The probabilities are stored in layersAfterActivationTest[v][1:layerSize[v]+1]
# 	maxxx=0.0
# 	answer=1



# for numberOfIterations in range (1):
# 	for i in range(int(n/batchSize)):
# 		p = i*batchSize
# 		for length in range(len(capitalTriangle)):
# 			capitalTriangle[length] = np.zeros([layerSizes[length+1],(layerSizes[length]+1)])
# 		for j in range(batchSize):
# 			layersActivation[0][0] = 1
# 			layersActivation[0][1:(m+1)] = x[j+p][:]
# 			for v in range(1,len(layerSizes)-1):
# 				layersActivation[v][0]=1
# 				layersActivation[v][1:] = activation(np.dot(weights[v-1],(layersActivation[v-1])),activationFunction)
# 			v = len(layerSizes)-1
# 			layersActivation[v][0]=1
# 			layersActivation[v][1:] = activation(np.dot(weights[v-1],(layersActivation[v-1])),"sigmoid")
# 			errorTerm[0] = np.zeros([layerSizes[0]+1]) #Error for first layer is 0
# 			errorTerm[len(layerSizes)-1][1:] = np.subtract((layersActivation[len(layerSizes)-1][1:], y[j+p])) #* (gradient(layersActivation[len(layerSizes)-1][1:],"sigmoid")) #Error for last layer
# 			for v in range(len(layerSizes)-2,0,-1):
# 				errorTerm[v] = np.multiply(np.dot((weights[v].transpose()),(errorTerm[v+1][1:])), gradient(layersActivation[v],activationFunction))
# 				capitalTriangle[v] = np.sum(capitalTriangle[v],np.outer(errorTerm[v+1][1:],layersActivation[v]))
# 			capitalTriangle[0] = np.sum(capitalTriangle[0], np.outer(errorTerm[1][1:],layersActivation[0]))	
# 			if i==0 and j==0:
# 				print(numberOfIterations)
# 		for length in range(len(weights)):		
# 			gradientOfCost[length] = np.sum((capitalTriangle[length], np.multiply(lambdaa,weights[length])))/batchSize
# 			weights[length] = np.subtract(weights[length], np.multiply(etaKnot, gradientOfCost[length])) #See how to change this

			

		# for j in range(batchSize):
		# 	layersActivation[0][0] = 1
		# 	layersActivation[0][1:(m+1)] = x[j+p][:]
		# 	for v in range(1,len(layerSizes)-1):
		# 		layersActivation[v][0]=1
		# 		layersActivation[v][1:] = activation(np.dot(weights[v-1],(layersActivation[v-1])),activationFunction)
		# 	v = len(layerSizes)-1
		# 	layersActivation[v][0]=1
		# 	layersActivation[v][1:] = activation(np.dot(weights[v-1],(layersActivation[v-1])),"sigmoid")
		# 	errorTerm[0] = np.zeros([layerSizes[0]+1]) #Error for first layer is 0
		# 	errorTerm[len(layerSizes)-1][1:] = (layersActivation[len(layerSizes)-1][1:] - y[j+p])
		# 	for v in range(len(layerSizes)-2,0,-1):
		# 		errorTerm[v] = np.dot((weights[v].transpose()),(errorTerm[v+1][1:])) * gradient(layersActivation[v],activationFunction)
		# 		capitalTriangle[v] = capitalTriangle[v] + np.outer(errorTerm[v+1][1:],layersActivation[v])
		# 	capitalTriangle[0] = capitalTriangle[0] + np.outer(errorTerm[1][1:],layersActivation[0])	
		# 	if i==0 and j==0:
		# 		print(numberOfIterations)

# def getRobertsPosFiltered(n,x):
# 	robertsPos = x[0:n][:]
# 	image = np.array([np.array([0 for e in range(32)]) for k in range(32)])
# 	# counter=0
# 	fig, ax = plt.subplots(ncols=2)
# 	for p in range(n):
# 		counter=0
# 		for e in range(32):
# 			image[e] = x[p][counter:counter+32]
# 			counter = counter+32
# 		ax[0].imshow(image, cmap=plt.cm.gray)	
# 		#plt.show()
# 		robertsPos_output = roberts_pos_diag(image)
# 		ax[1].imshow(robertsPos_output, cmap=plt.cm.gray)
# 		plt.show()
# 		count=0
# 		for e in range(32):
# 			robertsPos[p][count:count+32]=robertsPos_output[e]
			
# 	return robertsPos	

# def getRobertsNegFiltered(n,x):
# 	robertsNeg = x[0:n][:]
# 	image = np.array([np.array([0 for e in range(32)]) for k in range(32)])
# 	# counter=0
# 	fig, ax = plt.subplots(ncols=2)
# 	for p in range(n):
# 		counter=0
# 		for e in range(32):
# 			image[e] = x[p][counter:counter+32]
# 			counter = counter+32
# 		ax[0].imshow(image, cmap=plt.cm.gray)	
# 		#plt.show()
# 		robertsNeg_output = roberts_neg_diag(image)
# 		ax[1].imshow(robertsNeg_output, cmap=plt.cm.gray)
# 		plt.show()
# 		count=0
# 		for e in range(32):
# 			robertsNeg[p][count:count+32]=robertsNeg_output[e]
			
# 	return robertsNeg