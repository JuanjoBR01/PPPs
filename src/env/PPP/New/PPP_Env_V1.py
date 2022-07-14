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
 2. The agent won't make decisions according to a fixed (or random) policy, but according to the q_table
 3. The q values must be updated depending on the new state
 4. There are two types of maintenance, full and partial (not final)
'''

'''
Next steps:
    1. Let the agent make random decisions with the epsilon parameter
    2. Review Juan's library to change the inputs when creating the instance 
    3. Inspection to be decided randomly or coincidentally when the performance is a determinated value or less 
    4. Delete unused and redundant parameters (contained by the class)
    5. Keep improving the graphics

:)  0. Parametrize the number of discrete states for the performance. In this script is a fixed value = 5 
:)  0. Add a scatter plot for when the agent decides to fix
:)  0. Let the user decide whether see one replica of the model or the complete simulation for 1000 paths
'''

'''
Important details:
1. Currently, the agent con only make two decisions, fix or not. When fixing, the performance reaches 100%
2. Q_table is being initialized randomly
'''


# -------------------------------- lIBRARY IMPORT ---------------------------------

from math import exp, log
import matplotlib.pyplot as plt
import numpy as np
from random import random
plt.style.use('ggplot')



# -------------------------------- CLASS DEFINITION ---------------------------------

class EnvPPP():

    # -------------------- Builder Method --------------------

    def __init__(self):

        # --- Variable that lets the user to view a single replica or the complete simulation ---
        self.SINGLE_REPLICA = True


        # ---- Class Atributes ----
        self.S = [0, 1, 2]                          # Current state (for when running [Age, Performance, Budget]
        self.T = 30								    # Planning horizon
        self.W = [0, 1] 		                    # [shock?, inspection?] (currently deterministic)

        self.FC = 1                                 # Fixed maintenance cost
        self.VC = 1                                 # Variable maintenance cost    
        self.rate = 3 								# "Slope" of sigmoidal benefit-performance function
        self.offset = 3 							# "Offset" of sigmoidal benefit-performance function
        self.threshold = 0.6                        # Performance treshold (discrete)

        self.episodes = 30                          # Environment episodes

        # ---- Hyperparameters -----
        
        self.LEARNING_RATE = 0.01
        self.DISCOUNT = 0.95
        self.epsilon = 0.5         
        self.START_EPSILON_DECAYING = 1                                                                                  # Episode where epsilon begins to affect
        self.END_EPSILON_DECAYING = self.episodes // 2                                                                   # Episode where epsilon does not affect anymore
        self.epsilon_decay_value = self.epsilon/(self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING)                # Rate of decay of epsilon after each step
        self.NUM_INTERVALS = 100

        # ---- Q-Table ----
        self.q_table = np.random.uniform(low = -2, high = 0, size = ([self.NUM_INTERVALS] + [2]))   
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

    def discretize_performance(self, perf):
        # TODO Samuel: review the MIP - The bigger level, the better for JJ, otherwise for Samuel
        return int(min(max(np.ceil(perf*self.NUM_INTERVALS)-1, 0), self.NUM_INTERVALS-1))

    # Incentive calculated depending on the inspection

    def incentive(self, choice='sigmoid'):

        if self.W[1] == 0:
            return 0

        perf = self.S[1]
        if choice=='sigmoid':
            rate, offset = 10, self.threshold
            incent = 1/( 1 + exp(-rate*(perf-offset)))			

        elif choice=='linear':
            slope, offset = 1, 0
            incent = offset + slope*perf
        
        return incent

    # Function that deteriorates the system

    def deteriorate(self):
        age = self.S[0]		
        shock = self.W[0]

        Lambda = -log(self.threshold)/10   	# perf_tt = exp(-Lambda*tt) --> Lambda = -ln(perf_tt)/tt x
		
        if shock:
            delta_perf = 3*random()
        else:
            delta_perf = exp(-Lambda*age) - exp(-Lambda*(age-1))

        return delta_perf

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


    # Cost function
    def cost(self, X):
        return -self.FC*X - self.VC*X + 7*self.incentive()


    # Transition between states function
    def transition(self, X):
        delta_perf = self.deteriorate()
        bud = self.cost(X)

        if X:
            self.S = [0, 1, self.S[2]+bud]
        else:
            self.S = [self.S[0]+1, self.S[1]+delta_perf, self.S[2]+bud]
            

        return delta_perf, bud

    # Action function
    def fixed_action_rule_agent(self, random_exploration):
        # According to the q_table we'll make the decision
        perf = self.S[1]
        discrete_state = int(min(max(np.ceil(perf*self.NUM_INTERVALS)-1, 0), self.NUM_INTERVALS-1))

        if not random_exploration:
            return np.argmax(self.q_table[discrete_state])

        else:        
            return round(random())



    # Function to plot the important metrics of the environment

    def show(self, inspect, maint, perf):
        fig = plt.figure(figsize =(10, 5))
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

        Inspection = {t:(t,inspect[t]) for t in range(self.episodes) if inspect[t]}
        Maintenance = {t:(t,maint[t]) for t in range(self.episodes) if maint[t]}
        ax.plot(range(self.episodes), perf, 'k--', linewidth = 1.5, label = "Equilibrium solution for both parties")
        ax.plot([Inspection[t][0] for t in Inspection], [Inspection[t][1] for t in Inspection], 'rs', label = "Inspection actions")
        ax.plot([Maintenance[t][0] for t in Maintenance], [Maintenance[t][1] for t in Maintenance], 'b^', label = "Maintenance actions")
        ax.set_xlabel("Period", size=15)
        ax.set_ylabel("Road's Performance", size=15)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
        plt.suptitle("Leader's perspective" , fontsize=15)
        plt.grid(True)
        plt.savefig('Leader perspective.png')
        plt.show()
        # plt.close(fig)

	# Function to iterate the environment

    def run(self):
        performance = []
        cashflow = []
        inspections = []
        maintenances = []


        for episode in range(self.episodes):

            X = self.fixed_action_rule_agent(False)

            #self.W[1] = self.inspect('random_50', episode)
            #self.W[1] = self.inspect('reach_99', episode)
            self.W[1] = self.inspect('fixed_5', episode)

            inspections.append(self.W[1])
            maintenances.append(X)

            # Update of the q_table
            prev_state = int(min(max(np.ceil(self.S[1]*self.NUM_INTERVALS)-1, 0), self.NUM_INTERVALS-1))
            current_q = self.q_table[prev_state, X]

            values = self.transition(X)
            reward = values[1]

            new_state = int(min(max(np.ceil(self.S[1]*self.NUM_INTERVALS)-1, 0), self.NUM_INTERVALS-1))
            max_future_q = np.max(self.q_table[new_state])
            
            new_q = (1-self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q)
            
            self.q_table [prev_state, X] = new_q

            performance.append(self.S[1])
            cashflow.append(self.S[2])

        '''
        plt.subplot(2,1,1)
        plt.plot(performance)
        plt.subplot(2,1,2)
        plt.plot(cashflow)
        plt.show()
        '''

        if self.SINGLE_REPLICA:
            self.show(inspections, maintenances, performance)


        return cashflow


myPPP = EnvPPP()

if myPPP.SINGLE_REPLICA:
    # Single replica
    myPPP.run()

else:
    # MONTECARLO SIMULATION

    num_simulations = 1000
    periods = range(1, myPPP.episodes+1)

    fig = plt.figure()
    plt.title("PPP Simulation " + str(num_simulations) + " Paths")
    plt.xlabel("Period")
    plt.ylabel("Balance [$]")

    for i in range (num_simulations):
        myPPP = EnvPPP()
        cashflow = []
        cashflow = myPPP.run()
        plt.plot(periods, cashflow)


    plt.show()
