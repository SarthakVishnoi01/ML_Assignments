import csv
import numpy as np
import scipy as sc
import pandas as pd
import math
import sys
from numpy.linalg import inv
import time
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

def sigmoidGradient(x):
	x = np.array(x)
	return x*(1-x)
def reluGradient(x):
	x=np.array(x)
	if(x==0):
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

args = sys.argv

startTheCode = time.time()
file_read = r'devnagri_train.csv'
train = pd.read_csv(file_read,header=None)
file2 = r'devnagri_test_public.csv'
test = pd.read_csv(file2,header=None)	
test = test.values
train = train.values
y = train[:,0]
k=0
for i in range(len(y)):
	if(y[i]>k):
		k = y[i]
k = k+1
#print("This is k", k)

x = train[:,1:len(train[0])]
n = len(x); #Number of Examples
m = len(x[0]); #Number of Features
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
batchSize = 100 #int(args[4]) #100 #Take from command line
etaKnot = 0.8 #float(args[5]) #0.01 #take from command line
activationFunction = "sigmoid" #Take from command line
#hLayers = args[7:] #Taken from coomand line
#hiddenLayers = list(map(int,hLayers))
hiddenLayers = [500,300,100]
hiddenLayersSize = len(hiddenLayers)
layerSizes = [m]
for i in range(hiddenLayersSize):
	layerSizes.append(hiddenLayers[i])
layerSizes.append(k)
weights = np.array([np.array([np.array([(1.0*np.random.random_sample()-0.5) for p in range(layerSizes[i]+1)]) for j in range(layerSizes[i+1])]) for i in range(len(layerSizes)-1)]) 
layersActivation = np.array([np.array([np.array([(1.0*np.random.random_sample()-0.5) for e in range(batchSize)])for p in range(layerSizes[i]+1)]) for i in range(len(layerSizes))])
errorTerm = np.array([np.array([np.array([(1.0*np.random.random_sample()-0.5) for e in range(batchSize)])for p in range(layerSizes[i]+1)]) for i in range(len(layerSizes))])

gradientOfCost = np.array([np.array([np.array([0.0 for p in range(layerSizes[i]+1)]) for j in range(layerSizes[i+1])]) for i in range(len(layerSizes)-1)]) 
capitalTriangle = np.array([np.array([np.array([0.0 for p in range(layerSizes[i]+1)]) for j in range(layerSizes[i+1])]) for i in range(len(layerSizes)-1)])
#print(len(errorTerm))
inputData = np.array([np.array([0.0 for p in range(batchSize)]) for i in range(m+1)])
outputData = np.array([np.array([0.0 for p in range(batchSize)]) for i in range(k)])

#print(len(inputData))
#print(len(inputData[0]))

start = time.time()
#n = batchSize
for numberOfIterations in range (22):
	#print("Epoch number is", numberOfIterations)
	for i in range(int(n/batchSize)):
		p = i*batchSize
		capitalTriangle = capitalTriangle - capitalTriangle

		inputData[0][:] = np.array([1.0 for e in range(batchSize)])	
		inputData[1:][:] = (x[p:p+batchSize][:]).transpose() #1025*100

		outputData = (y[p:p+batchSize]).transpose() #46*100
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
		capitalTriangle[0] = capitalTriangle[0] + np.dot(errorTerm[1][1:],(layersActivation[0]).transpose())	

		gradientOfCost = (capitalTriangle + lambdaa*weights)/batchSize
		weights = weights - etaKnot * gradientOfCost #See how to change this

finish = time.time()
#print("Time taken is", (finish-start))

#np.savetxt("secondWeights.txt", weights[1], fmt="%f")
tempLayer = layersActivation[1]	
#np.savetxt("secondActivationLayer.txt", tempLayer, fmt="%f")
#print("Backpropogation Done!")
yTest = test[:,0]

xTest = test[:,1:len(test[0])]
nTest = len(xTest); #Number of Examples
#print("nTest is", nTest)
mTest = len(xTest[0]); #Number of Features
#print("mTest is", mTest)
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

#print("xTest", len(xTest))
#print("xTest[0]", len(xTest[0]))
#print("AfterAssignment Activation Layer", len(layersActivationTest[0]))

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
np.savetxt(args[3],yAnswer , fmt="%d")
#print("Atleast Compile toh hua!!")
