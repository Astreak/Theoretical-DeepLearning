from MacCullochPitts import *
import numpy as np

class Perceptron(MacCullochPittsNeuron):
        def __init__(self,inp):
                super(Perceptron,self).__init__(inp)
        # here threashold can be learnt
        def generate_weights(self):
                n=self.I.__len__()
                self.w=np.random.uniform(-0.01,-0.1,(n,))
        @property
        def cond(self):
                S=0
                for i in range(len(self.I)):
                        S+=self.I[i]*self.w[i];
                return S>=0
        def learning(self,real_pred):
                conv=False
                while not conv:
                        f=True
                        for i in range(len(self.I)):
                                ws=0;
                                for i,j in zip(self.I,self.w):
                                        ws+=i*j
                                if real_pred==1 and ws<0:
                                        self.w+=self.I
                                        f=False
                                elif real_pred==0 and ws>=0:
                                        self.w-=self.I
                                        f=False
                        conv=f
                return self.w;

                                
if __name__=="__main__":
        A=[3,0,0,1,1]
        per=Perceptron(A)
        per.generate_weights()
        W=per.learning(1)
        print(W)
        # S=0
        # for a,b in zip(A,W):
        #         S+=a*b
        # print(S)