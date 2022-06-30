class PPP_Ins():

    def __init__(self):
        
        # General settings
        self.card_T = 30
        self.card_S = 100
        
        # Sets
        self.L = range(7,0,-1)
        self.T = range(self.card_T)
        self.S = range(self.card_S)
        
        # Costs and earnings
        self.cf = [200/((1+0.1)**t) for t in self.T]
        self.cv = [150/((1+0.1)**t) for t in self.T]
        self.cg = [150/((1+0.1)**t) for t in self.T]
        self.cp = [150/((1+0.1)**t) for t in self.T]
        self.alpha = 5
        self.f = {}
        for t in self.T:
            if t in list(range(0,len(self.T),5)):
                self.f[t] = 100/((1+0.1)**t)
            else:
                self.f[t] = 0
        
        self.d = {}
        for l in self.L:
            aux = 10.0-2*l
            for t in self.T:
                self.d[(l,t)] = aux/((1+0.1)**(t))
                
        self.k = {}
        for l in self.L:
            aux = {5:-2, 4:-1, 3:.0, 2:1, 1:2}
            for t in self.T:
                #self.k[(l,t)] = aux[l]/((1+0.1)**(t))
                self.k[(l,t)] = 0
        
        # Scenarios
        self.prob = {s:1/len(self.S) for s in self.S}
                   
        # Minimum performance 
        self.minP = 0.65
        
        # Benefit and target profit
        self.epsilon = 1.0*100
        
        # Service-level ranges
        self.xi_L = {7:0 , 6:.11 ,5:.26, 4:.40, 3:.56, 2:.71, 1:.86}
        self.xi_U = {7:.10 , 6:.25 ,5:.40, 4:.55, 3:.70, 2:.85, 1:1}	

def importPPP():
    return(PPP_Ins())