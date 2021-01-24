import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

class SigmoidNeuron:
        def __init__(self,a):
                self.a=np.array(a).T
                self.eta=0.4
                self.w=np.random.uniform(-0.1,0.1,(self.a.shape[0],1))
                self.b=np.random.uniform(-0.1,0.1,(1,1))
        def sigmoid(self,x):
                return 1.0/(1.0+np.exp(-1*(np.dot(self.w.T,self.a))))
        def p_sigmoid(self,x):
                return self.sigmoid(x)*(1-self.sigmoid(x))
        def gradient_descent(self,y):
                y=np.array(y).T
                e=100
                P=[]
                for i in range(e):
                        pred=self.sigmoid(np.dot(self.w.T,self.a)+self.b)
                        error=0.5*(pred-y)**2
                        P.append(np.sum(error,axis=1))
                        p_error=pred-y
                        temp=p_error*self.p_sigmoid(np.dot(self.w.T,self.a)+self.b)
                        dw=np.dot(self.a,temp.T)
                        db=np.sum(temp,axis=1)
                        self.w=self.w-self.eta*dw
                        self.b=self.b-self.eta*db
                        
                        
                        
                plt.plot([i for i in range(e)],P,c='r')
                plt.show()
                # for i in P:
                #         print(i)
        def evaluate(self,x):
                x=np.array(x).T
                return self.sigmoid(np.dot(self.w.T,x)+self.b)
        
                
                
                
if __name__=="__main__":
        L=[[0,0],[0,1],[1,0],[1,1]]
        p=SigmoidNeuron(L)
        print("Before gradient descent: ")
        print(p.evaluate([[1,1]]).T)
        p.gradient_descent([[0],[0],[0],[1]])
        print("Prediction after GD")
        print(p.evaluate([[1,1]]).T)