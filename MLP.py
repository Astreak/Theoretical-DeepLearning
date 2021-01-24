import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tt
from sklearn.metrics import *
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.style.use('fivethirtyeight')

class SigmoidNeuron:
        def __init__(self,a):
                self.a=np.array(a).T
                self.eta=0.001
                self.w=np.random.uniform(-0.1,0.1,(self.a.shape[0],1))
                self.b=np.random.uniform(-0.1,0.1,(1,1))
                
        def sigmoid(self,x):
                return 1.0/(1.0+np.exp(-x))
        def p_sigmoid(self,x):
                return self.sigmoid(x)*(1-self.sigmoid(x))
        def gradient_descent(self,y):
                y=np.array(y).T
                e=500
                P=[]
                W=np.linspace(-10,10,500,dtype=np.float64)
                B=np.linspace(-10,10,500,dtype=np.float64)
                for i in range(e):
                        pred=self.sigmoid(np.dot(self.w.T,self.a)+self.b)
                        error=0.5*(pred-y)**2
                        P.append(np.mean(error,axis=1))
                        p_error=pred-y
                        temp=p_error*self.p_sigmoid(np.dot(self.w.T,self.a)+self.b)
                        dw=np.dot(self.a,temp.T)
                        db=np.sum(temp,axis=1)
                        self.w=self.w-self.eta*dw
                        self.b=self.b-self.eta*db
                
                A,B=np.meshgrid(W,B)
                fig=plt.figure()
                ax=fig.gca(projection='3d')
                ax.plot_surface(A,B,np.array(P), rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)
                plt.show()
                plt.savefig('Loss curve Gradient Descent')
                
                # for i in P:
                #         print(i)
        def evaluate(self,x):
                x=np.array(x).T
                
                return self.sigmoid(np.dot(self.w.T,x)+self.b)
        
                
                
                
if __name__=="__main__":
        data=pd.read_csv('class.csv')
        data=shuffle(data)
        X=data.iloc[:,2:-1].values
        y=data.iloc[:,-1].values.reshape(-1,1)
        sc=StandardScaler()
        X=sc.fit_transform(X)
        X_train,x_test,Y_train,y_test=tt(X,y,test_size=0.2)
        model=SigmoidNeuron(X_train)
        model.gradient_descent(Y_train)
        pred=model.evaluate(x_test)
        pred=np.squeeze(pred)
        for i in range(len(pred)):
                if pred[i]>0.487:
                        pred[i]=1
                else:
                        pred[i]=0
        
        c=0
        T=0
        y_test=np.squeeze(y_test)
        for i in range(len(pred)):
                if pred[i]==y_test[i]:
                        c+=1
                T+=1
        print(f'Test_case accuracy {(c/T)*100}')       # accuracy 85.7 to 90.3 depending on no of epochs