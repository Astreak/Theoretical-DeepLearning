import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import scipy
from sklearn.preprocessing import *
from sklearn.model_selection import *


class FeedForward(object):
        def __init__(self,shape):
                self.shape=shape
                self.w1=np.random.uniform(-0.01,0.01,(shape,2560))
                self.w2=np.random.uniform(-0.01,0.01,(2560,256))
                self.w3=np.random.uniform(-0.01,0.01,(256,64))
                self.w4=np.random.uniform(-0.01,0.01,(64,1))
                self.b1=np.zeros((2560,1))
                self.b2=np.zeros((256,1))
                self.b3=np.zeros((64,1))
                self.b4=np.zeros((1,1))
                self.lr=0.001;
                self.g=False
        def softmax(self,x):
                m=x.shape[1]
                n=x.shape[0]
                for i in range(m):
                        c=0
                        for j in range(n):
                                c+=x[j][i]
                        for j in range(n):
                                temp=np.exp(x[j][i])
                                x[j][i]=temp/c
                return x;
        def sigmoid(self,x):
                return 1.0/(1.0 + np.exp(x))
        def p_sigmoid(self,x):
                return self.sigmoid(x)*(1-self.sigmoid(x))
        def cross_entropy(self,y,ypred):
                return -1*y*np.log(ypred);
        def train(self,X,Y):
                e=100
                P=[]
                for i in range(e):
                        #forward propagation
                        o1=self.sigmoid(np.dot(self.w1.T,X)+self.b1) # 2560,m
                        o2=self.sigmoid(np.dot(self.w2.T,o1)+self.b2) # 256,m
                        o3=self.sigmoid(np.dot(self.w3.T,o2)+self.b3) # 64,m
                        o4=self.softmax(np.dot(self.w4.T,o3)+self.b4) #10,m
                        error=o4-Y
                        print(o4)
                        #backward propagation/ gradient descent
                        dw4=np.dot(o3,error.T)
                        db4=np.sum(error,axis=1)
                        h1=np.dot(self.w4,error)*self.p_sigmoid(o3)
                        dw3=np.dot(o2,h1.T)
                        db3=np.sum(h1,axis=1)
                        h2=np.dot(self.w3,h1)*self.p_sigmoid(o2)
                        dw2=np.dot(o1,h2.T)
                        db2=np.sum(h2,axis=1)
                        inp=np.dot(self.w2,h2)*self.p_sigmoid(o1)
                        dw1=np.dot(X,inp.T)
                        db1=np.sum(inp,axis=1)
                        self.w1=self.w1-self.lr*dw1
                        self.b1=self.b1-self.lr*db1
                        self.w2=self.w2-self.lr*dw2
                        self.b2=self.b2-self.lr*db2
                        self.w3=self.w3-self.lr*dw3
                        self.b3=self.b3-self.lr*db3
                        self.w4=self.w4-self.lr*dw4
                        self.b4=self.b4-self.lr*db4
                return;
        def test(self,x):
                if self.g==False:
                        assert "You have to train the model first otherwise whats the point of testing"
                o1=self.sigmoid(np.dot(self.w1.T,x)+self.b1) # 2560,m
                o2=self.sigmoid(np.dot(self.w2.T,o1)+self.b2) # 256,m
                o3=self.sigmoid(np.dot(self.w3.T,o2)+self.b3) # 64,m
                o4=self.softmax(np.dot(self.w4.T,o3),+self.b4) #10,m
                return o4

if __name__=="__main__":
        X=np.array([[1,2,3],[3,4,5],[5,6,7]])
        Y=np.array([[10],[20],[300]])
        ffn=FeedForward(X.shape[1])
        ffn.train(X.T,Y.T)        
                
        
                        
                        
                        
                        
                
                        
                
                
        
                                
                
        
                
                