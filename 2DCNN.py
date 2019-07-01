
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import os
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Conv3D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils
from keras.layers import Dropout, Input
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam


# In[2]:


dataset = pd.read_csv(r'C:\Users\slowg\Desktop\Datasetm\Complete_Data.csv')
print(dataset.shape)


# In[3]:


m,n = dataset.shape
X = dataset.iloc[:,0:n-1].values
y = dataset.iloc[:,-1].values
print(X)
print(X.shape)
print(y)
print(y.shape)


# In[4]:


X = X.reshape(145,145,200)
print(X.shape)


# In[5]:


y = y.reshape(145,145)


# In[6]:


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


# In[7]:


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca


# In[8]:


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


# In[9]:


def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels


# In[10]:


k = 30
from sklearn.decomposition import PCA
X,pca = applyPCA(X,numComponents=k)

X.shape


# In[11]:


windowSize = 5
numPCAcomponents = 30
testRatio = 0.30


# In[12]:



X, y = createImageCubes(X, y, windowSize=windowSize)

X.shape, y.shape


# In[13]:


Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, testRatio)

Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape


# In[14]:


Xtrain = Xtrain.reshape(-1, windowSize, windowSize, k)
Xtrain.shape


# In[15]:


print(ytrain)


# In[16]:


ytrain = np_utils.to_categorical(ytrain)
print(ytrain.shape)
print(Xtrain.shape)


# In[17]:


input_shape= Xtrain[0].shape
print(input_shape)


# In[18]:


C1 = 3*numPCAcomponents


# In[19]:


model = Sequential()

model.add(Conv2D(C1, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(3*C1, (3, 3), activation='relu'))
model.add(Dropout(0.25))



model.add(Flatten())
model.add(Dense(6*numPCAcomponents, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='softmax'))


# In[20]:


adam = Adam(lr=0.001, decay=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# In[21]:


model.fit(Xtrain, ytrain, batch_size=32, epochs=15)


# In[24]:



ytest = np_utils.to_categorical(ytest)
test_loss, test_acc = model.evaluate(Xtest, ytest)
print('Test accuracy:', test_acc)

