'''
 --------------------- PUBLIC PRIVATE PARTNERSHIPS ENVIRONMENT V1 ---------------------
    A principal-agent setting for the manteinance of a deteriorating system
 ------------------------------- Research Team ---------------------------------------
        * Camilo Gómez
        * Samuel Rodríguez 
        * Juan José Beltrán Ruiz 
        * Juan Betancourt
        * Martín Romero

'''

'''
Next steps:
    1. Let the agent make random decisions with the epsilon parameter
    2. Review Juan's library to change the inputs when creating the instance 
    3. Keep improving the graphics
    4. Review why the random seed is affecting the behaviour 
    5. Change the budget to be the cashfow of each period
'''

'''
Important details:
1. Currently, the agent con only make two decisions, fix or not. When fixing, the performance reaches 100%
2. Q_table is being initialized randomly
3. In this case, the state variable is the number of days without fixing
'''


# -------------------------------- lIBRARY IMPORT ---------------------------------

from math import exp, log
import matplotlib.pyplot as plt
import numpy as np
import random

plt.style.use('ggplot')

np.random.seed(10)
random.seed(10)

# -------------------------------- CLASS DEFINITION ---------------------------------

class EnvPPP():

    # -------------------- Builder Method --------------------

    def __init__(self):

        # ---- Class Atributes ----
        self.S = [0, 1, 2]                          # Current state (for when running [Age, Performance, Budget]
        self.T = 30								    # Planning horizon
        self.W = [0, 1] 		                    # [shock?, inspection?
        self.NUM_INTERVALS = 5
        self.L = range(1,self.NUM_INTERVALS + 1)	# discrete levels of performance


        self.FC = 1                                 # Fixed maintenance cost
        self.VC = 1                                 # Variable maintenance cost    
        self.rate = 3 								# "Slope" of sigmoidal benefit-performance function
        self.offset = 3 							# "Offset" of sigmoidal benefit-performance function
        self.threshold = 0.6                        # Performance treshold (discrete)


        self.fail = .2										# failure threshold (continuous)
        self.ttf = 10.0										# time to failure
		
        self.Lambda = -log(self.fail)/self.ttf				# perf_tt = exp(-Lambda*tt) --> Lambda = -ln(perf_tt)/tt
		
		
        self.f_W = [(1, [False, False])] 					# scenarios: (prob_s, [shock, inspection])

        self.gamma = .9										# discount factor
        self.v_hat = {i:0 for i in range(int(self.ttf))} 	# state value estimation

        self.alpha = .5		

        
        #Social benefit related to the performance level. g_star is the expected one 

        self.g = {1:2, 2:47, 3:500, 4:953, 5:998}
        # Earnings target
        self.g_star = 595

        self.theta = [exp(-self.Lambda*tau) for tau in range(self.T+2)]

        self.bond = {}
        for level in self.L:
            average_l = 0
            count_l = 0
            for gamma_val in self.theta:
                if self.get_level(gamma_val) == level:
                    average_l += 7*self.incentive(gamma_val) 
                    count_l += 1
            self.bond[level] = average_l/count_l

	

    def get_level(self, perf):
        if perf < .2:
            return 1
        elif perf < .4:
            return 2
        elif perf < .6:
            return 3
        elif perf < .8:
            return 4
        else:
            return 5

    # Incentive calculated depending on the inspection    
    def MIP_incentive(self, dwm):
        #if self.W[1] == 0:
        #    return 0
        
        return self.bond[self.get_level(dwm)]


    def incentive(self, perf, choice='sigmoid'):

        # Samuel has discretized the function according to the performance level
        #if self.W[1] == 0:
        #    return 0

        if choice=='sigmoid':
            rate, offset = 10, self.threshold
            incent = 1/( 1 + exp(-rate*(perf-offset)))			


        elif choice=='linear':
            slope, offset = 1, 0
            incent = offset + slope*perf
        
        return incent

    # Function to decide when will the government inspect the project
    def inspect(self, policy, episode):

        # Three options: fixed (every 5 periods), random_x (bernoulli probability), reach (if the performance gets to a level)

        if policy[:5] == 'fixed':
            aux = episode + 1
            return 1 if episode > 0 and aux % int(policy[6:]) == 0 else 0

        elif policy[:6] == 'random':
            return round( random() < int(policy[7:])/100.0)
        
        elif policy[:5] == 'reach':
            level = int(policy[6:]) / 100
            return 1 if self.S[1] <= level else 0


    # Cost function depending on the incentive
    def cost(self, dwm, X):

        perf = self.theta[dwm]
        # The fisrt thing is to obtain the incentive depending on the performance. It is calculated with the days that have passed wothout maintenance
        # dwm refers to days without maintenance

        #return -self.FC*X - self.VC*X + 7*self.incentive()
        #return -self.FC*X - self.VC*X + 7*self.MIP_incentive()
        return -self.FC*X - self.VC*X + self.MIP_incentive(dwm) 


    # Transition between states function
    def transition(self, dwm, X):
        if X:
            return 0
        else:
            return dwm + 1


    # Action function
    def fixed_action_rule_agent(self, random_exploration):
        # According to the q_table we'll make the decision
        perf = self.S[1]
        #discrete_state = int(min(max(np.ceil(perf*self.NUM_INTERVALS)-1, 0), self.NUM_INTERVALS-1))
        discrete_state = self.get_level(perf) - 1

        if not random_exploration:
            return np.argmax(self.q_table[discrete_state])
        elif self.S[2] < self.FC - self.VC:
            return 0
        else:        
            return np.random.randint(2)


	
    '''
		Bellman's recursion for a GIVEN policy
	'''
    def V(self, pi, S):		

        SM, C = self.transition, self.cost 		# for short		

        if type(pi)==dict:
            x = pi[S]
        else:
            x = pi(S)

        value = sum( w[0]*(C(S, x) + self.gamma*self.v_hat[ SM(S, x) ] ) for w in self.f_W )
        return value


    '''
		Bellman's recursion to find a policy
	'''
    def V_bw(self, S, arg):		

        SM, C = self.transition, self.cost 		# for short

        def Ew_Bellman(x):
            return sum( w[0]*( C(S, x) + self.gamma*self.v_hat[ SM(S, x) ] ) for w in self.f_W )
		
        cand = [(x, Ew_Bellman(x)) for x in self.X]

        x_best, value = min(cand, key = lambda h: h[1])
		
        if arg:
            return x_best
        else:
            return value


    
    '''
		policy iteration
	'''
    def policy_iteration(self):
        self.v_hat = {i:0 for i in range(int(self.ttf))}
        policy_is_changing = True
        current_policy = {i:0 for i in range(int(self.ttf))}

        while policy_is_changing:

			# Cycle to EVALUATE current policy
            while True:
                error = 0
                for i in range(int(self.ttf)):
                    old_val = self.v_hat[i]
                    self.v_hat[i] = self.V(current_policy, i)
                    error = max(error, abs(self.v_hat[i]-old_val) )
                if error < 1e-12:
                    break
			
			# Cycle to UPDATE current policy
            policy_is_changing = False
            for i in range(int(self.ttf)):
                old_action = current_policy[i]
                current_policy[i] = self.V_bw(i, True) 	# True: returns x_best
													# The trick: V_bw uses a new 'v_hat'
                if old_action != current_policy[i]:
                    policy_is_changing = True
        return current_policy



# ----------------------- DECLARING, INITIALIZING AND RUNNING POLICY ITERATION -----------------------

# Declaration of the instance
myPPP = EnvPPP()

myPPP.policy_iteration()
