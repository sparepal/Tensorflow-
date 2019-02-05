# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:03:37 2018

@author: vikhy
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class hs():
    def __init__(self,train,test):
        self.train=train
        self.test=test
        
class Train():
    def __init__(self,images,labels):
        self.images=images
        self.labels=labels
    def next_batch(self,batch_size=15):
        idx = np.arange(0 , len(self.images))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        images_shuffle = [self.images[ i] for i in idx]
        labels_shuffle = [self.labels[ i] for i in idx]
        return (np.asarray(images_shuffle), np.asarray(labels_shuffle))

class Test():            
    def __init__(self,images,labels):
        self.images=images
        self.labels=labels

def load_split_scale_data(train_fn='sign_mnist_train.csv',test_fn='sign_mnist_test.csv'):
    #Loading
    train=pd.read_csv(train_fn)
    test=pd.read_csv(test_fn)
    #Splitting
    X_train= train.iloc[:,1:]
    y_train=train.iloc[:,0]
    X_test=test.iloc[:,1:]
    y_test=test.iloc[:,0]
    #Normalize pixel values
    scaler=MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #Encode the labels using one hot encoder
    labelencoder_y = LabelEncoder()
    y_train = labelencoder_y.fit_transform(y_train[:])
    y_test = labelencoder_y.fit_transform(y_test[:])
    
    onehotencoder = OneHotEncoder(categorical_features = [0])
    y_test = onehotencoder.fit_transform(y_test.reshape(-1,1)).toarray()
    y_train = onehotencoder.fit_transform(y_train.reshape(-1,1)).toarray()

    return hs(Train(X_train,y_train),Test(X_test,y_test))







