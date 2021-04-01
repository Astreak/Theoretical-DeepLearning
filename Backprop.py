import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import scipy
from sklearn.preprocessing import *
from sklearn.model_selection import *
np.seterr(over='ignore')
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
                self.w1=np.random.uniform(-0.01,0.01,(shape,25))
                self.w2=np.random.uniform(-0.01,0.01,(25,25))
                self.w3=np.random.uniform(-0.01,0.01,(shape,14))
                self.w4=np.random.uniform(-0.01,0.01,(14,1))
                self.b1=np.zeros((25,1))
                self.b2=np.zeros((25,1))
                self.b3=np.zeros((14,1))
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
                e=1000
                P=[]
                for i in range(e):
                        #forward propagation
                        # a1=np.dot(self.w1.T,X) +self.b1
                        # o1=self.sigmoid(a1)
                        # a2=np.dot(self.w2.T,o1) +self.b2
                        # o2=self.sigmoid(a2)
                        a3=np.dot(self.w3.T,X) +self.b3
                        o3=self.sigmoid(a3)
                        a4=np.dot(self.w4.T,o3) +self.b4
                        o4=self.sigmoid(a4)
                        error=o4-Y
                        P.append(np.sum(error))
                        error=self.p_sigmoid(error)
                        dw4=np.dot(o3,error.T)
                        db4=np.mean(error,axis=1)
                        db4=np.expand_dims(db4,axis=1)
                        # 
                        h1=np.dot(self.w4,error)*self.p_sigmoid(a3)
                        dw3=np.dot(X,h1.T)
                        db3=np.mean(h1,axis=1)
                        db3=np.expand_dims(db3,axis=1)
                        # 
                        # h2=np.dot(self.w3,h1)*self.p_sigmoid(a2)
                        # dw2=np.dot(o1,h2.T)
                        # db2=np.mean(h2,axis=1)
                        # db2=np.expand_dims(db2,axis=1)
                        # 
                        # h3=np.dot(self.w2,h2)*self.p_sigmoid(a1)
                        # dw1=np.dot(X,h3.T)
                        # db1=np.mean(h3,axis=1)
                        # db1=np.expand_dims(db1,axis=1)
                        # self.w1-=self.lr*dw1
                        # self.b1-=self.lr*db1
                        # self.w2-=self.lr*dw2
                        # self.b2-=self.lr*db2
                        self.w3-=self.lr*dw3
                        self.b3-=self.lr*db3
                        self.w4-=self.lr*dw4
                        self.b4-=self.lr*db4
                                                

                plt.plot([i for i in range(e)],P,c='c')
                plt.show()
                print("ok")
                self.g=True
                return;
        def test(self,x):
                if self.g==False:
                        assert "You have to train the model first otherwise whats the point of testing"
                # o1=self.sigmoid(np.dot(self.w1.T,x)+self.b1) # 2560,m
                # o2=self.sigmoid(np.dot(self.w2.T,o1)+self.b2) # 256,m
                # o3=self.sigmoid(np.dot(self.w3.T,o2)+self.b3) # 64,m
                o4=self.sigmoid(np.dot(self.w4.T,x)+self.b4) #10,m
                return o4

if __name__=="__main__":
        Train,Test=CreateDataset('class.csv').split()
        X_train,Y_train=Train
        x_test,y_test=Test
        ffn=FeedForward(X_train.shape[1])
        ffn.train(X_train.T,Y_train.T)
        print(ffn.test(x_test.T).T)
        
           
                        
                        
                        
                        
                
                        
                
                
        
                                
                
        
                
                