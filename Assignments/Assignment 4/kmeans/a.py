
# coding: utf-8

# In[30]:


import numpy as np
import scipy
from scipy.stats import norm
import math
import random
import time


# In[61]:


train = np.load('DHC_full_train.npy')
test = np.load('DHC_full_test.npy')


# In[62]:


print(len(train))
print(len(train[0]))
print(len(test))
print(len(test[0]))


# In[33]:


np.random.shuffle(train)
print(train[63773][0])


# In[34]:


xTrain = train[:,1:]
xTrain = xTrain/255
print(xTrain.shape)
yTrain = train[:,0]
print(yTrain.shape)


# In[63]:


xTest = test[:,:]
xTest = xTest/255
print(xTest.shape)


# In[36]:


output = []
for x in yTrain:
    if x not in output:
        output.append(x)
print(output)


# In[37]:


print(len(output))


# In[38]:


k = len(output)
d = len(xTrain[0])
n = len(xTrain)
print(k, d, n)


# In[42]:


class mykMeans:

    def __init__(self,k,n,d):
        # k is the number of clusters to form
        self.k = k
        # centroid points
        self.centroid = np.zeros(k*d).reshape(k,d)
        # d is the dimension of the feature vector
        self.d = d
        # n is the number of the samples we have
        self.n = n
        #cluster is the stored value of the data
        self.cluster = np.zeros(n)
        #mapping from maxPoints in cluster to cluster label
        self.mapping = np.zeros(k)

    def showValues(self):
        print("k:",self.k)
        print("n:",self.n)
        print("d:",self.d)

    def dis(self,data,centroidIndex):
#         distance =0
#         using euclidean distance with weights in this case
#         for i in range (0,self.d):
#             distance = distance + (self.weights[i]*((data[i]-self.centroid[centroidIndex][i])**2))
        dist = np.linalg.norm(data-self.centroid[centroidIndex])
        return dist

#     def updateClass(self,data):
# #         print('n in update class: ',self.n)
#         for i in range (0,self.n):
#             minDistance = math.inf
#             if(i%100 == 0):
#                 print("Update Class:", i)
# #             print("i in updateClass: ",i)
#             for j in range (0,self.k):
#                 tempDist = self.dis(data[i],j)
#                 if(tempDist<minDistance):
#                     self.cluster[i] = j
#                     minDistance = tempDist
                    # print('cluster: ',j)
        # self.printAllClasses()
    def updateClass(self,data):
        for i in range(self.n):
            if(i%10000 == 0):
                print("Update Class:", i)
            minDistance = math.inf
            point = data[i]
            mat = np.array([point,] * self.k)
            distArray = np.linalg.norm(mat - self.centroid, axis=1)
            minIndex = np.argmin(distArray)
            self.cluster[i] = minIndex

    def updateCentroids(self,data):
        converged = True
        tempCentroid = np.zeros(self.k*self.d).reshape(self.k,self.d)
        tempCnumbers = np.zeros(self.k)
        for i in range(0,self.n):
            clusIndex = (int)(self.cluster[i])
            # print('clusindex for i:',i,clusIndex)
            tempCnumbers[clusIndex]+=1
            tempCentroid[clusIndex]+=data[i]
        for i in range(0,self.k):
            if(tempCnumbers[i] != 0):
                t = (tempCentroid[i]/tempCnumbers[i])==self.centroid[i]
            else:
                t = tempCentroid[i] == self.centroid[i]
#             print("K=", i)
            for j in range(0,self.d):
                if(t[j]==False):
                    converged = False
                    break
            if(tempCnumbers[i] != 0):        
                self.centroid[i] = tempCentroid[i]/tempCnumbers[i]
            else:
                self.centroid[i] = tempCentroid[i]
        return converged

    def initializeCentroids(self,data):
        #right now it takes only the first k elements
        for i in range(0,self.k):
            self.centroid[i] = data[random.randint(0,self.n-1)]
    
    def initKMeansPlusPlus(self,data):
        ##KMeans++ implementation
        self.centroid[0] = data[random.randint(0,self.n-1)]
        for i in range(1,self.k):
            distArr = []
            for j in range(self.n):
                mindist = math.inf
                for k in range(i):
                    tempDist = self.dis(data[j],k)
                    if mindist>tempDist:
                        mindist=tempDist
                distArr.append(mindist)
            distArr = np.array(distArr)
            probs = distArr/(np.sum(distArr))
            cumprobs = np.cumsum(probs)
            r = scipy.rand()
            for s,t in enumerate(cumprobs):
                if r<t:
                    self.centroid[i] = data[s]
                    break
            
                
    def classify(self,data,max_iter,y):
#         self.initKMeansPlusPlus(data)
        convergence = False
        self.initializeCentroids(data)
        p=0
        while((convergence==False) and (p!=max_iter)):
            start = time.time()
            print(p)
            p+=1
            self.updateClass(data)
            convergence = self.updateCentroids(data)
            finish = time.time()
            print("Time for", p,"th iteration is:", finish-start, "seconds")
        print("Classification Done")
        self.getCorrectLabelsForClusters(y)
#         print("Now printing cluster values:")
#         self.printAllClasses()

    def getClass(self,data):
        minDistance = math.inf
        clusterNumber = -1
#         point = data
#         mat = np.array([point,] * self.k)
#         distArray = np.linalg.norm(mat - self.centroid, axis=1)
#         clusterNumber = np.argmin(distArray)
        for j in range (0,self.k):
            tempDist = np.linalg.norm(data - self.centroid[j])
            if(tempDist<minDistance):
                clusterNumber = j
                minDistance = tempDist
        return self.mapping[clusterNumber]
#         return clusterNumber
    
    def getClassForAll(self,data):
        n = data.shape[0]
        ansList=[]
        for i in range(n):
            ans = self.getClass(data[i])
            ansList.append(ans)
#             print(ans)
        return ansList
        
    def printAllClasses(self):
        for i in range(0,self.n):
            print(self.cluster[i])
            
    def getCorrectLabelsForClusters(self,y):
        k = 46
        conversionMatrix = np.zeros(self.k*k).reshape(self.k,k)
#         print(self.cluster)
        for i in range(self.n):
            conversionMatrix[int(self.cluster[i])][y[i]] += 1
#             print(conversionMatrix)
        self.mapping = np.argmax(conversionMatrix, axis=1)
        print(self.mapping)
#         return labelCluster
            


# In[52]:


obj = mykMeans(800,n,d)


# In[53]:


start = time.time()
obj.classify(xTrain,4,yTrain)
finish = time.time()
print("Total time taken for prediction is:", finish-start, "seconds")


# In[54]:


# obj.getCorrectLabelsForClusters(yTrain)
output = []
for x in obj.mapping:
    if x not in output:
        output.append(x)
print(output)


# In[55]:


cll = np.zeros([len(output)])
for i in range(len(obj.mapping)):
    pp = obj.mapping[i]
    cll[pp] += 1
print(cll)


# In[56]:


#Accuracy Calculation
start = time.time()
prediction = obj.getClassForAll(xTrain)
finish = time.time()
print(finish - start)
# start = time.time()
boolArray = (prediction == yTrain)
count = np.sum(boolArray)
# finish = time.time()
# print(finish-start)
# count = 0
# start = time.time()
# for i in range(len(prediction)):
#     if(prediction[i] == yTrain[i]):
#         count += 1
# finish = time.time()
# print(finish-start)
print("Accuracy is", 100*count/len(prediction),"%")

