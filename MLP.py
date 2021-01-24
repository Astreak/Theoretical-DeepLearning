import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

class SigmoidNeuron:
        def __init__(self,a):
                self.a=np.array(a)
                self.eta=0.29
                self.w=np.random.uniform(-0.1,0.1,(len(a),))
                self.b=np.random.uniform(-0.1,0.1,(1,))
        def sigmoid(self,x):
                return 1.0/(1.0+np.exp(-1*(np.dot(self.w.T,self.a))))
        def p_sigmoid(self,x):
                return self.sigmoid(x)*(1-self.sigmoid(x))
        def gradient_descent(self,y):
                e=1000
                P=[]
                for i in range(e):
                        pred=self.sigmoid(np.dot(self.w.T,self.a)+self.b)
                        error=0.5*(pred-y)**2;
                        P.append(error)
                        p_error=(pred-y)
                        temp=p_error*self.p_sigmoid(pred)*self.a
                        h=p_error*self.p_sigmoid(pred)
                        self.w=self.w-self.eta*temp
                        self.b=self.b-self.eta*h
                # plt.plot([i for i in range(e)],P,c='r')
                # plt.show()
                # for i in P:
                #         print(i)
        def evaluate(self,x):
                x=np.array(x)
                return self.sigmoid(np.dot(self.w.T,x)+self.b)
        
                
                
                
if __name__=="__main__":
        L=[0.11,0.53]
        p=SigmoidNeuron(L)
        print("Before gradient descent: ")
        print(p.evaluate([0.10,0.51]))
        p.gradient_descent(1)
        print("Prediction after GD")
        print(p.evaluate([0.10,0.51]))