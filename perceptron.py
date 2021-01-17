from MacCullochPitts import *

class Perceptron(MacCullochPittsNeuron):
        def __init__(self,inp):
                super(Perceptron,self).__init__(inp)
        def show(self):
                return self.I

if __name__=="__main__":
        print(Perceptron([12,34]).show())