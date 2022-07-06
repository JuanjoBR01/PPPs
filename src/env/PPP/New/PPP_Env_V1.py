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
 Changes compared to the original code
 1. Creation of Q_Table
 2. The agent won't make decisions according to a fixed policy, but according to the q_table
 3. The q values must be updated depending on the new state
 4. There are two types of maintenance, full and partial (not final)
'''

'''
Next steps:
1. Let the agent make random decisions with the epsilon parameter
2. Parametrize the number of discrete states for the performance. In this script is a fixed value = 5
3. Review Juan's library to change the inputs when creating the instance 
'''


# 1. lIBRARY IMPORT

from math import exp, log
from random import random
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


# 2. HYPERPARAMETERS 
LEARNING_RATE = 0.01
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 500

epsilon = 0.5                                                                                               # Probability to make a random exploration
START_EPSILON_DECAYING = 1                                                                                  # Episode where epsilon begins to affect
END_EPSILON_DECAYING = EPISODES // 2                                                                        # Episode where epsilon does not affect anymore
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)                               # Rate of decay of epsilon after each step

NUM_INTERVALS = 100


# 3. CLASS DEFINITION

class EnvPPP():

    # Function that initializes the environment

    def __init__(self):
        self.S = [0,0.35,2]                         # Current state (for when running [Age, Performance, Budget]
        self.X = [0, 1]								# Available actions
        self.L = range(NUM_INTERVALS)               # Discrete levels
        self.T = 40								    # Planning horizon
        self.W = [False, True] 		                # [shock?, inspection?] (currently deterministic)

        self.FC = 10                                # Fixed maintenance cost
        self.VC = 1                                 # Variable maintenance cost    
        self.rate = 3 								# "Slope" of sigmoidal benefit-performance function
        self.offset = 3 							# "Offset" of sigmoidal benefit-performance function
        self.threshold = 0.6                        # Performance treshold (discrete)

        self.episodes = 3000                          # Environment episodes
        self.q_table = np.random.uniform(low = -2, high = 0, size = ([NUM_INTERVALS] + [2]))


        '''
        Deterioration cycles model
        self.t_reach_half = 12
        self.Q = range(self.t_reach_half)
        '''

        '''
        Necessary parmeters to model the deterioration
        '''

        self.fail = .2										# failure threshold (continuous)
        self.ttf = 10.0										# time to failure
		
        self.Lambda = -log(self.fail)/self.ttf				# perf_tt = exp(-Lambda*tt) --> Lambda = -ln(perf_tt)/tt
		

        '''
        Return's epsilon
        Benefit and target profit
        self.epsilon = 1.4*100

        Parameter f (¿?)
        TODO: Review exact paper   
        self.f ={0:0,1:135,2:0,3:0,4:0,5:83.8243786129859,6:0,7:0,8:0,9:0,10:52.0483440729868,11:0,12:0,13:0,14:0,15:32.3179266648371,16:0,17:0,18:0,19:0,20:20.0668897832594,21:0,22:0,23:0,24:0,25:12.4599597539036,26:0,27:0,28:0,29:0,30:7.73665469565768}

        VMT modeling - Discount Factor
        self.d = {}
        for i in range(1,6):
            b = 10.0-2*i
            for j in range(1,31):
                self.d[(i,j)]=(b/(1+0.1)**(j-1))
		

        # Minimum performance 
        self.minP = .6
        '''
		
        self.f_W = [(1, [False, False])] 					# scenarios: (prob_s, [shock, inspection])

        self.gamma = .9										# discount factor
        self.v_hat = {i:0 for i in range(int(self.ttf))} 	# state value estimation

        self.alpha = .5		

        '''
        Social benefit related to the performance level. g_star is the expected one 
        '''
        self.g = {5:2, 4:47, 3:500, 2:953, 1:998}
        # Earnings target
        self.g_star = 595

        '''
        Fixed income from the principal to the agent
        # self.a = 50
        '''


        '''
        Leader budget
        self.Beta = 5e8
        '''

        '''
        Do we want to use the bond or the sigmoidal function?
                bond = {1: {5:-29.65, 4:-27.4, 3:-4.75, 2:17.9, 1:20.15},
                2:{5:-148.25, 4:-137, 3:-23.75, 2:89.5, 1:100.75},
                3:{5:-296.5, 4:-274, 3:-47.5, 2:179, 1:201.5},
                4:{5:-444.75, 4:-411, 3:-71.25, 2:268.5, 1:302.25}}

        1 is the default
        self.bond = bond[1]

        '''

        '''
        Fixed cost of inspection
        self.c_sup_i = 1
        c_sup_i = {1:50, 2:250, 3:70}
        self.c_sup_i = c_sup_i[INS]
        '''
	


    # Function that returns the incentive according to the performance level

    def discretize_performance(perf):
        # TODO Samuel: review the MIP
        return int(min(max(np.ceil(perf*NUM_INTERVALS)-1, 0),NUM_INTERVALS-1))

    def incentive(self, S, W, choice='sigmoid'):
        perf = S[1]
        if choice=='sigmoid':
            rate, offset = 10, self.threshold
            incent = 1/( 1 + exp(-rate*(perf-offset)))			

        elif choice=='linear':
            slope, offset = 1, 0
            incent = offset + slope*perf
        
        return incent
        


    # Function that deteriorates the system

    def deteriorate(self, S, W):
        age = S[0]		
        shock = W[0]

        Lambda = -log(self.threshold)/10   	# perf_tt = exp(-Lambda*tt) --> Lambda = -ln(perf_tt)/tt x
		
        if shock:
            delta_perf = 3*random()
        else:
            delta_perf = exp(-Lambda*age) - exp(-Lambda*(age-1))

        return delta_perf

    # Cost function
    def cost(self, S, X, W, episode):
        # Inspect each 5 episodes
        return -self.FC*X - self.VC*X + 7*self.incentive(S, W) if not episode % 5 else -self.FC*X - self.VC*X

    # Transition between states function
    def transition(self, S, X, W, episode):
        if X:
            self.S[0] = 0
            self.S[1] = 1
        delta_perf = self.deteriorate(S, W)
        bud = self.cost(S, X, W, episode)

        self.S = [S[0]+1, S[1]+delta_perf, S[2]+bud]

        return delta_perf, bud

    # Action function
    def fixed_action_rule_agent(self, q_table, random_exploration):
        # According to the q_table we'll make the decision
        perf = self.S[1]
        discrete_state = int(min(max(np.ceil(perf*NUM_INTERVALS)-1, 0),NUM_INTERVALS-1))

        if not random_exploration:
            print(np.argmax(q_table[discrete_state]))
            return np.argmax(q_table[discrete_state])

        else:        
            return round(random())

	# Function to iterate the environment

    def run(self):
        performance = []
        cashflow = []

        print(self.q_table)

        for episode in range(self.episodes):	

            X = self.fixed_action_rule_agent(self.q_table, False)

            # Update of the q_table
            prev_state = int(min(max(np.ceil(self.S[1]*NUM_INTERVALS)-1, 0),NUM_INTERVALS-1))
            prev_performance = self.S[1]
            current_q = self.q_table[prev_state, X]

            values = self.transition(self.S, X, self.W, episode)
            delta_perf = values[0]
            reward = values[1]

            new_state = int(min(max(np.ceil(self.S[1]*NUM_INTERVALS)-1, 0),NUM_INTERVALS-1))
            max_future_q = np.max(self.q_table[new_state])
            
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            
            self.q_table [prev_state, X] = new_q

            performance.append(self.S[1])
            cashflow.append(self.S[2])

        plt.subplot(2,1,1)
        plt.plot(performance)
        plt.subplot(2,1,2)
        plt.plot(cashflow)
        plt.show()


myPPP = EnvPPP()
myPPP.run()














































