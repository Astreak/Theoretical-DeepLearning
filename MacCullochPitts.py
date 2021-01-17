class MacCullochPittsNeuron(object):
        def __init__(self,inp):
                self.I=inp
                self.output=[0]*(len(self.I))
        # threashold cant be learned
        def logic(self,thres):
                self.S=0
                for i in self.I:
                     self.S+=i
                return self.S>=thres
        def function(self):
                return self.I;

if __name__=="__main__":
        pass
        # A=[0,1]
        # mc=MacCullochPittsNeuron(A)
        # print(mc.logic(1))
        
                
                