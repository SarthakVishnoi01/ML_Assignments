
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import sys
import time


# In[16]:


#argument handling

train = pd.read_csv(sys.argv[1],header=0).values
valid = pd.read_csv(sys.argv[2],header=0).values
test = pd.read_csv(sys.argv[3],header=0).values
output = sys.argv[4]
plot = sys.argv[5]


# In[4]:


# part = sys.argv[1]
# file1 = r'train.csv'
# train = pd.read_csv(file1)
# train = train.values
np.random.shuffle(train)

# file2 = r'valid.csv'
# valid = pd.read_csv(file2)
# valid = valid.values
np.random.shuffle(valid)

# file3 = r'test.csv'
# test = pd.read_csv(file3)
# test = test.values

print(len(train))
print(len(train[0]))
print(len(test))
print(len(test[0]))
print(test)


# In[5]:


print(train[0])


# In[6]:


le = LabelEncoder()
for i in range(len(train[0])):
#     print(i)
    if(isinstance(train[0][i], str)):
        train[:,i] = le.fit_transform(train[:,i])
        valid[:,i] = le.fit_transform(valid[:,i])
        test[:,i] = le.fit_transform(test[:,i])


# In[7]:


print(train[0])


# In[8]:


#preparation of train, test and validation data
xTrain = train[:,1:]
yTrain = train[:,0].astype('int')
xTest = test[:,1:]
xValid = valid[:,1:]
yValid = valid[:,0].astype('int')


# In[12]:


obj = DecisionTreeClassifier(criterion="entropy",splitter='best',random_state=0,max_depth=12,min_samples_leaf=30)
obj.fit(xTrain,yTrain)
yPred = obj.predict(xTest)
np.savetxt(output, yPred, fmt="%d")


# In[13]:


nodes = np.arange(2,200,1)
trainList = np.zeros(len(nodes))
validList = np.zeros(len(nodes))
pointer = 0
for i in range(len(nodes)):
    if(i%100 == 0):
        print(i)
    tempObj = DecisionTreeClassifier(criterion="entropy",splitter='best',max_leaf_nodes=nodes[i])
    tempObj.fit(xTrain,yTrain)
    yTrainTemp = tempObj.predict(xTrain)
    yValidTemp = tempObj.predict(xValid)
    p = accuracy_score(yTrain,yTrainTemp)*100
    q = accuracy_score(yValid,yValidTemp)*100
    trainList[pointer] = p
    validList[pointer] = q
    pointer += 1


# In[15]:


plt.plot(nodes, trainList, linestyle='-', color='r')
plt.plot(nodes, validList, linestyle='-', color='b')
plt.title('part b')
plt.xlabel('no. of nodes in decision tree')
plt.ylabel('accuracy in %')
plt.legend(['train set', 'validation set'])
plt.savefig(plot)

