import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import scipy
from sklearn.preprocessing import *
from sklearn.model_selection import *

class CreateDataset(object):
        def __init__(self,name):
                self.data=pd.read_csv(name)
                self.X=self.data.iloc[:,2:-1].values
                self.Y=self.data.iloc[:,-1].values.reshape(-1,1)
                self.sc=StandardScaler()
                self.X=self.sc.fit_transform(self.X)
        def split(self):
                X_train,x_test,Y_train,y_test=train_test_split(self.X,self.Y,test_size=0.2,random_state=0)
                return (X_train,Y_train),(x_test,y_test)
        def random(self):
                return;

                
                
                
                
                
class FeedForward(object):
        def __init__(self,shape):
                self.shape=shape
                self.w1=np.random.uniform(-0.1,0.1,(shape,2560))
                self.w2=np.random.uniform(-0.1,0.1,(2560,256))
                self.w3=np.random.uniform(-0.1,0.1,(256,64))
                self.w4=np.random.uniform(-0.1,0.1,(64,1))
                self.b1=np.zeros((2560,1))
                self.b2=np.zeros((256,1))
                self.b3=np.zeros((64,1))
                self.b4=np.zeros((1,1))
                self.lr=0.01;
                self.g=False
        def softmax(self,x):
                ex=np.exp(x-np.max(x))
                return ex/ex.sum(axis=1)
        def sigmoid(self,x):
                return 1.0/(1.0 + np.exp(-1*x))
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
                        u=np.sum(error)
                        P.append(u)
                        #backward propagation/ gradient descent
                        dw4=np.dot(o3,error.T)
                        db4=np.sum(error,axis=1)
                        db4=np.expand_dims(db4,axis=1)
                
                        h1=np.dot(self.w4,error)*self.p_sigmoid(o3)
                        dw3=np.dot(o2,h1.T)
                        db3=np.sum(h1,axis=1)
                        db3=np.expand_dims(db3,axis=1)
                        h2=np.dot(self.w3,h1)*self.p_sigmoid(o2)
                        dw2=np.dot(o1,h2.T)
                        db2=np.sum(h2,axis=1)
                        db2=np.expand_dims(db2,axis=1)
                        inp=np.dot(self.w2,h2)*self.p_sigmoid(o1)
                        dw1=np.dot(X,inp.T)
                        db1=np.sum(inp,axis=1)
                        db1=np.expand_dims(db1,axis=1)

                        
                        self.w1=self.w1-self.lr*dw1
                        self.b1=self.b1-self.lr*db1
                        self.w2=self.w2-self.lr*dw2
                        self.b2=self.b2-self.lr*db2
                        self.w3=self.w3-self.lr*dw3
                        self.b3=self.b3-self.lr*db3
                        self.w4=self.w4-self.lr*dw4
                        self.b4=self.b4-self.lr*db4
                plt.plot([i for i in range(e)],P,c='c')
                return;
        def test(self,x):
                if self.g==False:
                        assert "You have to train the model first otherwise whats the point of testing"
                o1=self.sigmoid(np.dot(self.w1.T,x)+self.b1) # 2560,m
                o2=self.sigmoid(np.dot(self.w2.T,o1)+self.b2) # 256,m
                o3=self.sigmoid(np.dot(self.w3.T,o2)+self.b3) # 64,m
                o4=self.softmax(np.dot(self.w4.T,o3)+self.b4) #10,m
                return o4

if __name__=="__main__":
        Train,Test=CreateDataset('class.csv').split()
        X_train,Y_train=Train
        x_test,y_test=Test
        ffn=FeedForward(X_train.shape[1])
        ffn.train(X_train.T,Y_train.T)
        pred=ffn.test(x_test.T)
        print(pred.T)      
                        
                        
                        
                        
                
                        
                
                
        
                                
                
        
                
                