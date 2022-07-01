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
'''

'''
Next steps:
1. Let the agent make random decisions with the epsilon parameter
2. Parametrize the number of discrete states for the performance. In this script is a fixed value = 5
'''


# 1. lIBRARY IMPORT

from math import exp, log
from random import random
from this import d
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

NUM_INTERVALS = 20


# 3. CLASS DEFINITION

class EnvPPP():

    # Function that initializes the environment

    def __init__(self):
        self.S = [0,0.3,2]                          # Current state (for when running [Age, Performance, Budget]
        self.X = [0, 1]								# Available actions
        self.L = range(NUM_INTERVALS)               # Discrete levels
        self.T = 40								    # Planning horizon
        self.W = [False, True] 		                # [shock?, inspection?] (currently deterministic)

        self.FC = 10                                # Fixed maintenance cost
        self.VC = 1                                 # Variable maintenance cost    
        self.rate = 3 								# "Slope" of sigmoidal benefit-performance function
        self.offset = 3 							# "Offset" of sigmoidal benefit-performance function
        self.threshold = 0.6                        # Performance treshold

        self.episodes = 10000                     # Environment episodes
        self.q_table = np.random.uniform(low = -2, high = 0, size = ([NUM_INTERVALS] + [2]))


    # Function that returns the incentive according to the performance level

    def discretize_performance(perf):
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
    def cost(self, S, X, W):
        return -self.FC*X - self.VC*X + 7*self.incentive(S, W)

    # Transition between states function
    def transition(self, S, X, W):
        if X:
            self.S[0] = 0
            self.S[1] = min(self.S[1] + 0.3, 1)
        delta_perf = self.deteriorate(S, W)
        bud = self.cost(S, X, W)

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

            values = self.transition(self.S, X, self.W)
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














































