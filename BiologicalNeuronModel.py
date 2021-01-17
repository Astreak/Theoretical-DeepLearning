import numpy as np
import math
from collections import defaultdict
class BioNeuron(object):
        def __init__(self,no):
                self.no=no
                self.__synapse=defaultdict(list)
        @property
        def initialize_neurons(self):
                self.__L=[]
                for i in range(1,self.no+1):
                        self.__L.append(i)
        @property ## dendrite functionity
        def data_recieve(self):
                pass
        def conn(self,a,b):
                if a<=self.no and b<=self.no and a!=b:
                        self.__synapse[a].append(b)
                        self.__synapse[b].append(a)
                else:
                        raise Exception("Fuck off")
        @property
        def show_graph(self):
                for i in self.__synapse.keys():
                        print(i,end=" ");print(self.__synapse[i])
                

if __name__=="__main__":
        basicnu=BioNeuron(12)
        basicnu.initialize_neurons
        basicnu.conn(1,2)
        basicnu.conn(4,1)
        basicnu.show_graph
        