from math import log, exp

class PPP_Ins():

    def __init__(self,
                INC = 2,
                INS = 2):
        
        # General settings
        self.card_T = 30
        self.t_reach_half = 12
        self.card_S = 100
        
        # Penalty/Reward set
        #self.K = {i:i*1000 for i in range(-2,2)}
        #Bonds

        bond = {1: {5:-29.65, 4:-27.4, 3:-4.75, 2:17.9, 1:20.15},
                2:{5:-148.25, 4:-137, 3:-23.75, 2:89.5, 1:100.75},
                3:{5:-296.5, 4:-274, 3:-47.5, 2:179, 1:201.5},
                4:{5:-444.75, 4:-411, 3:-71.25, 2:268.5, 1:302.25}}

        '''bond = {1:{5:-4444.8, 4:-4107.65, 3:-713.35, 2:2680.95, 1:3018.1},
                                        2:{5:-22224, 4:-20538.25, 3:-3566.75, 2:13404.75, 1:15090.5},
                                        3:{5:-44448, 4:-41076.5, 3:-7133.5, 2:26809.5, 1:30181},
                                        4:{5:-66672, 4:-61614.75, 3:-10700.25, 2:40214.25, 1:45271.5}} #1500 multiplier'''

        ''' bond = {1:{5:-1481.6, 4:-1369.25, 3:-237.8, 2:893.65, 1:1006},
                    2:{5:-7408, 4:-6846.25, 3:-1189, 2:4468.25, 1:5030},
                    3:{5:-14816, 4:-13692.5, 3:-2378, 2:8936.5, 1:10060},
                    4:{5:-22224, 4:-20538.75, 3:-3567, 2:13404.75, 1:15090}} #500 multiplier'''
    
        '''bond = {1:{5:-296.3, 4:-273.85, 3:-47.55, 2:178.75, 1:201.2},
                    2:{5:-1481.5, 4:-1369.25, 3:-237.75, 2:893.75, 1:1006},
                    3:{5:-2963, 4:-2738.5, 3:-475.5, 2:1787.5, 1:2012},
                    4:{5:-4444.5, 4:-4107.75, 3:-713.25, 2:2681.25, 1:3018}} #100 multiplier'''

        '''bond = {1:{5:-148.2, 4:-136.95, 3:-23.8, 2:89.35, 1:100.6},
                    2:{5:-741, 4:-684.75, 3:-119, 2:446.75, 1:503},
                    3:{5:-1482, 4:-1369.5, 3:-238, 2:893.5, 1:1006},
                    4:{5:-2223, 4:-2054.25, 3:-357, 2:1340.25, 1:1509}} #50 multiplier'''
        
        self.bond = bond[INC]
        
        # Sets
        self.L = {5:5,4:4,3:3,2:2,1:1}
        self.T = range(self.card_T)
        self.Q = range(self.t_reach_half)
        
        # Costs and earnings
        #self.cf = [round(150/((1+0.1)**tauu),2) for tauu in self.T]	#Fix Costs
        #self.cv = [round(150/((1+0.1)**tauu),2) for tauu in self.T]	#Variable Costs
        self.cf = [15 for tauu in self.T] #Fix Costs
        self.cv = [15 for tauu in self.T]    #Variable Costs
        self.alpha = 0
        self.f ={0:0,1:135,2:0,3:0,4:0,5:83.8243786129859,6:0,7:0,8:0,9:0,10:52.0483440729868,11:0,12:0,13:0,14:0,15:32.3179266648371,16:0,17:0,18:0,19:0,20:20.0668897832594,21:0,22:0,23:0,24:0,25:12.4599597539036,26:0,27:0,28:0,29:0,30:7.73665469565768}
        
        # Earnings Per performance level
        #self.g = {5:0,4:10*100,3:20*100,2:30*100,1:40*100}
        # Earnings target
        #self.g_star = 3000
        
        self.g = {5:2, 4:47, 3:500, 2:953, 1:998}
        # Earnings target
        self.g_star = 595

        # Fixed income from the principal to the agent
        self.a = 50
        
        # Leader Budget
        self.Beta = 5e8
        
        # Fixed cost of inspection
        #self.c_sup_i = 1
        c_sup_i = {1:50, 2:250, 3:70}
        self.c_sup_i = c_sup_i[INS]
        
        self.d = {}
        for i in range(1,6):
            b = 10.0-2*i
            for j in range(1,31):
                self.d[(i,j)]=(b/(1+0.1)**(j-1))
                
        self.k = {}
        self.k = {t:0 for t in self.T}
        
        # Deterioration
        self.lev_fail = .25
        self.det = -log(self.lev_fail)/len(self.T)
        self.gamma = [round(exp(-self.det*tau),2) for tau in self.T]
            
        # Minimum performance 
        self.minP = .6
        
        # Benefit and target profit
        self.epsilon = 1.4*100
        
        # Service-level ranges
        self.xi_L = {5:0, 4:.21, 3:.41, 2:.61, 1:.81}
        self.xi_U = {5:.2, 4:.4, 3:.6, 2:.8, 1:1}	

        print("\nThis is INC: " +str(INC))

        print("\nThis is bond: " + str(bond[INC]))
        
        print("\nThis is INS: " +str(INS))
        
        print("\nThis is c_sup_i: " + str(c_sup_i[INS]))

def importPPP():
    return(PPP_Ins())