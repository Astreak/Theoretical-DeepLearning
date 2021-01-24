import numpy as np
import pandas as pd
import random
import math

class MLP(object):
        def __init__(self,a,**k):
                self.a=a;
                self.k=k;
                self.w=np.random.uniform(-0.1,0.1,(a.shape[0],1))
                self.eta=0.13
        def helper(self,x):
                S=0
                for i in x:
                        S+=i
                return S;
        def sigmoid(self):
                return 1.0/(1.0+np.exp(-np.dot(self.w.T,self.a)))
        def P_sigmoid(self):
                return self.sigmoid*(1-self.sigmoid(self.a))


if __name__=="__main__":
        L=np.array([[0,0],[0,1],[1,0],[1,1]])
        p=MLP(L.T)
        

