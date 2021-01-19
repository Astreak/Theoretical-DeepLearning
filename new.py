import torch
from torch.nn import *
import torch.nn.functional as F
import torchvision
from torchvision import *
import torch.optim  as opt


class Model(Module):
        def __init__(self,shape):
                super(Model,self).__init__()
                self.l1=Linear(shape,64)
                self.l2=Linear(64,64)
                self.l3=Linear(64,64)
                self.l4=Linear(64,10)
        def forward(self,x):
                x=F.relu(self.l1(x))
                x=F.relu(self.l2(x))
                x=F.relu(self.l3(x))
                x=F.log_softmax(self.l4(x),dim=1)
                return x
        
# class GenerateDataset()  

class BuildNetwork(object):
        def __init__(self):
                self.data=[]
        def load_model(self):
                self.Train=datasets.MNIST("",download=True,train=True,transform=transforms.Compose([transforms.ToTensor()]));
                self.Test=datasets.MNIST("",download=True,train=False,transform=transforms.Compose([transforms.ToTensor()]));
        def ufo_dist(self):
                self.train=torch.utils.data.DataLoader(self.Train,batch_size=10,shuffle=True)
                self.test=torch.utils.data.DataLoader(self.Test,batch_size=10,shuffle=True)
        def init_model(self):
                self.load_model()
                self.ufo_dist()
                self.neural=Model(28*28)
        def train_model(self):
                self.init_model()
                self.optimizer=opt.Adam(self.neural.parameters(),lr=0.001)
                self.epoch=3
                for e in range(self.epoch):
                        for i,j in self.train:
                                self.neural.zero_grad()
                                X=i.view((-1,28*28))
                                pred=self.neural(X)
                                loss=F.nll_loss(pred,j)
                                loss.backward()
                                self.optimizer.step()
                        print(f"Loss function after end of epoch {e+1} : {loss}")
        def evaluate(self):
                c=0
                T=0
                for a,b in self.test:
                        pred=self.neural(a.view((-1,28*28)))
                        pred=torch.argmax(pred,axis=1)
                        
                        for i in range(len(pred)):
                                if b[i]==pred[i]:
                                        c+=1
                                T+=1
                print(f"Test case Accuracy is {(c/T)*100}")
        def run_model(self):
                self.train_model()
                print("Testing.....")
                self.evaluate()
                

if __name__=="__main__": 
        build=BuildNetwork()
        build.run_model()