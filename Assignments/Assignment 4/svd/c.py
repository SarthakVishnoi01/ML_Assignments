
# coding: utf-8

# In[18]:


import csv
import numpy as np
import scipy as sc
import pandas as pd
import math
import sys
from numpy import linalg
from numpy.linalg import inv
from decimal import Decimal
import cvxopt
from cvxopt import matrix, solvers
import time as time
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from matplotlib import pyplot as plt
from numpy import cov
args = sys.argv

file1 = r'DHC_train.csv'
train = pd.read_csv(file1,header=None)
train = train.values
np.random.shuffle(train)

file2 = r'DHC_test.csv'
test = pd.read_csv(file2,header=None)
test = test.values

print(len(train))
print(len(train[0]))
print(len(test))
print(len(test[0]))
print(train[29][345])
print(test)


# In[19]:


# testT = test.T
# testT = testT[1:]
print(len(test))
xTest = test[:,1:]
print(len(xTest))

# xTest = xTest/255
print(xTest.shape)
# y = train[0]
# np.savetxt("k.txt", k, fmt="%f")


# In[20]:


y = np.array([-1,1,-1,1,-1,1,-1])
x = np.array([[1,2],[4,7],[1,-1],[3,4],[0,1],[8,2],[-2,5]])
test = np.array([[10,4],[0,-2],[5,16],[-2,7]])

print(x)
k = x.sum(axis = 0)
k = k/len(x)
print("Printing k.....")
print(k)
x = x - k
print(x)


# In[21]:


# trainT = train.T
# trainT = trainT[1:]
xTrain = train[:,1:]
# xTrain = xTrain/255
n, p = xTrain.shape
print(n,p)
print(xTrain.shape)
# y = train[0]
yTrain = train[:,0]
yTrain = yTrain - 1*(yTrain==0)
print(yTrain.shape)
# xTrain = xTrain/255
m = len(xTrain)
n = len(xTrain[0])
print(m,n)
answers = np.loadtxt('answer.txt', dtype=int)
answers = answers - 1*(answers==0)


# In[5]:


top = 1024
M = np.mean(xTrain.T.tolist(), axis = 1)
M1 = np.mean(xTest.T.tolist(), axis = 1)

C = xTrain - M
C1 = xTest - M1

V = np.cov(C.T)
V1 = np.cov(C1.T)

values, vectors = np.linalg.eig(V)
values1, vectors1 = np.linalg.eig(V1)

print(vectors.shape)

vectors = vectors[:np.shape(vectors)[0], 0:top]
vectors1 = vectors1[:np.shape(vectors1)[0], 0:top]

xTrainFinal = vectors.T.dot(C.T)
xTrainFinal = xTrainFinal.T
xTestFinal = vectors1.T.dot(C1.T)
xTestFinal = xTestFinal.T

print(xTrainFinal.shape)
print(xTestFinal.shape)


# In[45]:


top = 50
xx = cov(xTrain.T)
u, s, vh = linalg.svd(xx)
pcaTrain = np.dot(xTrain, u[:,:top])/255
pcaTest = np.dot(xTest, u[:,:top])/255
print(pcaTrain.shape)
print(pcaTest.shape)


# In[46]:


count = 0
yTrain = yTrain - 1*(yTrain==0)


# In[47]:


checkX = np.array([[1,2,3,4],[6,3,9,0],[1,2,9,7],[1,3,5,7],[5,4,9,6]])


# In[48]:


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


# In[49]:


xTrainFinal.shape


# In[50]:


class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # print(kernel)
        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        print(self.C)
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples),'d')
        b = cvxopt.matrix(0.0)
        if self.C is None:      # hard-margin SVM
           G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
           h = cvxopt.matrix(np.zeros(n_samples))
        else:              # soft-margin SVM
           G = cvxopt.matrix(np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples))))
           h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        print ("Printing a")
        np.savetxt("beforeDeletionA.txt", a, fmt="%f")    
        print(a)
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("printing a")
        np.savetxt("a.txt", self.a, fmt="%f")    
        print("printing sv")
        np.savetxt("sv.txt", self.sv, fmt="%f")    
        print("printing W")
        np.savetxt("svy.txt", self.sv_y, fmt="%f")    

        print(self.a, self.sv, self.sv_y)
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
#         print("printing W")
#         np.savetxt("weights.txt", self.w, fmt="%f")    

    def project(self, X):
        print(self.b)
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))


# In[55]:


print(yTrain.shape)
print(pcaTrain.shape)
print(pcaTest.shape)


# In[56]:


clf = SVM(kernel=gaussian_kernel, C=1.0)

startTime = time.time()
print("Time start")
clf.fit(pcaTrain,yTrain)
print("Fitting Done")
endTime = time.time()
print(endTime-startTime)


# In[57]:


prediction = clf.predict(pcaTest)
print("Prediction done")
np.savetxt("cs1160336.txt", prediction, fmt="%d")
count = 0
for i in range(len(prediction)):
    if(prediction[i] == answers[i]):
        count += 1
print(len(prediction))
print("Accuracy=", count/len(prediction))


# In[59]:


print(len(prediction))
print(count)


# In[60]:


from sklearn import decomposition
pca = decomposition.PCA(n_components=50, whiten=True)
pca.fit(xTrain)


# In[63]:


plt.imshow(pca.mean_.reshape(32,32), cmap=plt.cm.bone)

