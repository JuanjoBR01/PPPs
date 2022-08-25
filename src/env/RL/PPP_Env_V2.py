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
'''


# -------------------------------- lIBRARY IMPORT ---------------------------------


from ctypes.wintypes import RGB
from math import exp, log
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import random
plt.style.use('ggplot')
from gurobipy import *
np.random.seed(10)
random.seed(10)

# -------------------------------- CLASS DEFINITION ---------------------------------


class EnvPPP():

    # -------------------- Builder Method --------------------

    def __init__(self):

        # ---- Class Atributes ----
        # Current state (for when running [Age, Performance, Budget]
        self.S = [0, 1, 2]
        self.T = 30								    # Planning horizon
        self.W = [0, 1] 		                    # [shock?, inspection?]
        self.NUM_INTERVALS = 5
        # discrete levels of performance
        self.L = range(1, self.NUM_INTERVALS + 1)

        self.FC = 1                                 # Fixed maintenance cost
        self.VC = 1                                 # Variable maintenance cost
        self.rate = 3 								# "Slope" of sigmoidal benefit-performance function
        self.offset = 3 							# "Offset" of sigmoidal benefit-performance function
        # Performance treshold (discrete)
        self.threshold = 0.6

        # ---- Hyperparameters -----

        self.LEARNING_RATE = 0.1
        self.DISCOUNT = 0.9
        self.epsilon = 0.15
        # Episode where epsilon begins to affect
        self.START_EPSILON_DECAYING = 1
        # Episode where epsilon does not affect anymore
        self.END_EPSILON_DECAYING = self.T // 2
        # Rate of decay of epsilon after each step
        self.epsilon_decay_value = self.epsilon / \
            (self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING)

        # ---- Q-Table ----
        # self.q_table = np.random.uniform(low = -2, high = 0, size = ([self.NUM_INTERVALS] + [2]))
        self.q_table = np.random.uniform(
            low=-35, high=-34.5, size=([self.NUM_INTERVALS] + [2]))

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

        # perf_tt = exp(-Lambda*tt) --> Lambda = -ln(perf_tt)/tt
        self.Lambda = -log(self.fail)/self.ttf

        '''
        Return's epsilon
        Benefit and target profit
        self.epsilon = 1.4*100

        Parameter f (¿?)
        TODO: Review exact paper
        self.f ={0:0,1:135,2:0,3:0,4:0,5:83.8243786129859,6:0,7:0,8:0,9:0,10:52.0483440729868,11:0,12:0,13:0,14:0,15:32.3179266648371,
            16:0,17:0,18:0,19:0,20:20.0668897832594,21:0,22:0,23:0,24:0,25:12.4599597539036,26:0,27:0,28:0,29:0,30:7.73665469565768}

        VMT modeling - Discount Factor
        self.d = {}
        for i in range(1,6):
            b = 10.0-2*i
            for j in range(1,31):
                self.d[(i,j)]=(b/(1+0.1)**(j-1))


        # Minimum performance
        self.minP = .6
        '''

        # scenarios: (prob_s, [shock, inspection])
        self.f_W = [(1, [False, False])]

        self.gamma = .9										# discount factor
        # state value estimation
        self.v_hat = {i: 0 for i in range(int(self.ttf))}

        self.alpha = .5

        # Social benefit related to the performance level. g_star is the expected one
        self.g = {1: 2, 2: 47, 3: 500, 4: 953, 5: 998}
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
        Bond according to the scenario
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

        self.gamma = [exp(-self.Lambda*tau) for tau in range(self.T+2)]

        self.bond = {}
        for level in self.L:
            average_l = 0
            count_l = 0
            for gamma_val in self.gamma:
                if self.get_level(gamma_val) == level:
                    average_l += 7*self.incentive(gamma_val)
                    count_l += 1
            self.bond[level] = average_l/count_l

    def reset(self):

        return [0, 1, 2]

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

    # Function that returns the incentive according to the performance level

    def discretize_performance(self, perf):
        # TODO Samuel: review the MIP - The bigger level, the better for JJ, otherwise for Samuel
        # return int(min(max(np.ceil(perf*self.NUM_INTERVALS)-1, 0), self.NUM_INTERVALS-1))
        return self.get_level(perf) - 1

    # Incentive calculated depending on the inspection

    def MIP_incentive(self):
        if self.W[1] == 0:
            return 0

        return self.bond[self.get_level(self.S[1])]

    def incentive(self, perf, choice='sigmoid'):

        # Samuel has discretized the function according to the performance level
        if self.W[1] == 0:
            return 0

        if choice == 'sigmoid':
            rate, offset = 10, self.threshold
            incent = 1/(1 + exp(-rate*(perf-offset)))

        elif choice == 'linear':
            slope, offset = 1, 0
            incent = offset + slope*perf

        return incent

    '''
    # Function that deteriorates the system
    def deteriorate(self):
        age = self.S[0]
        shock = self.W[0]

        # Lambda = -log(self.threshold)/10   	# perf_tt = exp(-Lambda*tt) --> Lambda = -ln(perf_tt)/tt x

        if shock:
            delta_perf = 3*random()
        else:
            delta_perf = exp(-self.Lambda*age) - exp(-self.Lambda*(age-1))

        return delta_perf
    '''

    # Function to decide when will the government inspect the project
    def inspect(self, policy, episode):

        # Three options: fixed (every 5 periods), random_x (bernoulli probability), reach (if the performance gets to a level)

        if policy[:5] == 'fixed':
            aux = episode + 1
            return 1 if episode > 0 and aux % int(policy[6:]) == 0 else 0

        elif policy[:6] == 'random':
            aux = random.uniform(0,100)
            return 1 if aux < int(policy[7:]) else 0

        elif policy[:5] == 'reach':
            level = int(policy[6:]) / 100
            return 1 if self.S[1] <= level else 0

    # Cost function depending on the incentive

    def cost(self, X):
        # return -self.FC*X - self.VC*X + 7*self.incentive()
        # return -self.FC*X - self.VC*X + 7*self.MIP_incentive()
        return -self.FC*X - self.VC*X + self.MIP_incentive()

    # Transition between states function

    def transition(self, X):
        # delta_perf = self.deteriorate()
        bud = self.cost(X)

        if X:
            self.S = [0, 1, self.S[2]+bud]
        else:

            self.S[0] += 1
            self.S[1] = self.gamma[self.S[0]+1]
            self.S[2] += bud

        return bud

    # Action function
    def fixed_action_rule_agent(self, random_exploration):

        # discrete_state = int(min(max(np.ceil(perf*self.NUM_INTERVALS)-1, 0), self.NUM_INTERVALS-1))
        discrete_state = self.get_level(self.S[1]) - 1

        if self.S[2] < self.FC + self.VC:
            return 0

        if not random_exploration:
            return np.argmax(self.q_table[discrete_state])
        else:
            return np.random.randint(2)

    # Function to plot the important metrics of the environment

    def show_performance(self, inspect, maint, perf):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

        Inspection = {t: (t, inspect[t]) for t in range(self.T) if inspect[t]}
        Maintenance = {t: (t, maint[t]) for t in range(self.T) if maint[t]}
        ax.plot(range(self.T), perf, 'k--', linewidth=1.5,
                label="Performance level")
        ax.plot([Inspection[t][0] for t in Inspection], [Inspection[t][1]
                for t in Inspection], 'rs', label="Inspection actions")
        ax.plot([Maintenance[t][0] for t in Maintenance], [Maintenance[t][1]
                for t in Maintenance], 'b^', label="Maintenance actions")
        ax.set_xlabel("Period", size=15)
        ax.set_ylabel("Road's Performance", size=15)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
        plt.suptitle("Leader's perspective", fontsize=15)
        plt.grid(True)
        plt.savefig('Performance.png')
        plt.show()

    # Function to plot the historic cashflow
    def show_budget(self, cashflow):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

        ax.plot(range(self.T), cashflow, 'k--',
                linewidth=1.5, label="Agent's Budget")
        ax.set_xlabel("Period", size=15)
        ax.set_ylabel("Cashflow ($)", size=15)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
        plt.suptitle("Leader's perspective", fontsize=15)
        plt.grid(True)
        plt.savefig('Cashflow.png')
        plt.show()

    def show_cashflows(self, cashflows):

        pos = []
        neg = []

        for i in range(self.T):
            if cashflows[i] < 0:
                neg.append(cashflows[i])
                pos.append(0)
            elif cashflows[i] > 0:
                pos.append(cashflows[i])    
                neg.append(0)
            else:
                pos.append(0)
                neg.append(0)
 
        fig = plt.subplots(figsize =(8, 5))
        p1 = plt.bar(range(self.T), pos, width=0.35, color = 'green')
        p2 = plt.bar(range(self.T), neg, width=0.35,bottom = pos, color = 'red')
 
        plt.ylabel('Cashflow ($)')
        plt.xlabel('Period')
        plt.title("Agent's Cashflows")

        plt.show()

    # Function to iterate the environment
    def run(self, state):
        # performance = [state[1]]
        # cashflow = [state[2]]
        performance = []
        cashflow = []
        inspections = []
        maintenances = []
        cashflow = []
        budget = []

        for episode in range(self.T):

            # Decide if a random exploration is done
            if np.random.random() > self.epsilon:
                X = self.fixed_action_rule_agent(False)
            else:
                X = self.fixed_action_rule_agent(True)

            #self.W[1] = self.inspect('random_10', episode)
            #self.W[1] = self.inspect('reach_50', episode)
            self.W[1] = self.inspect('fixed_5', episode)

            inspections.append(self.W[1])
            maintenances.append(X)

            # Update of the q_table
            # prev_state = int(min(max(np.ceil(self.S[1]*self.NUM_INTERVALS)-1, 0), self.NUM_INTERVALS-1))
            prev_state = self.get_level(self.S[1]) - 1
            current_q = self.q_table[prev_state, X]

            reward = self.transition(X)

            # new_state = int(min(max(np.ceil(self.S[1]*self.NUM_INTERVALS)-1, 0), self.NUM_INTERVALS-1))
            new_state = self.get_level(self.S[1]) - 1
            max_future_q = np.max(self.q_table[new_state])

            new_q = (1-self.LEARNING_RATE) * current_q + \
                self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q)

            self.q_table[prev_state, X] = new_q

            if episode == 0:
                performance.append(self.S[1])
                budget.append(self.S[2])
                cashflow.append(round(reward, 5))
            else:
                performance.append(self.S[1])
                budget.append(self.S[2])
                cashflow.append(round(reward, 5))

        return budget, inspections, maintenances, performance, cashflow


# ----------------------- DECLARING, INITIALIZING AND RUNNING THE ENVIRONMENT -----------------------

# Declaration of the instance
myPPP = EnvPPP()

# Declaration of the cumulative q_table, in the next replica, the agent will take the q_table that the previous agent left
new_q_table = myPPP.q_table

# Number of simulations to be run. The self.T will be ran the times in the parameter
num_simulations = int(1e3)

for i in range(num_simulations):
    state = myPPP.reset()

    myPPP.S = state

    if i == num_simulations-1:
        myPPP.epsilon = 0

    myPPP.q_table = new_q_table
    values = []
    values = myPPP.run(state)

    new_q_table = myPPP.q_table


# The order of the values array is [cashflow(0), inspections(1), maintenances(2), performance(3), reward -or cashflow- (4)]
#myPPP.show_performance(values[1], values[2], values[3])
#myPPP.show_budget(values[0])
#myPPP.show_cashflows(values[4])


# Section of the code that prints the decisions made by the agent at the end of the simulation among 30 periods

dictMaints = {}
dictInspections = {}
for i in range(len(values[2])):
    dictMaints['x_' + str(i)] = values[2][i]
    dictInspections['q_' + str(i)] = values[1][i]





def follower_PPP(x_param, maintenances):
    print(x_param)
    gamma = myPPP.gamma
    


    fc = [myPPP.FC for _ in range(myPPP.T)]
    vc = [myPPP.VC for _ in range(myPPP.T)]

    xi_L = {1:0, 2:.21, 3:.41, 4:.61, 5:.81}
    xi_U = {1:.2, 2:.4, 3:.6, 4:.8, 5:1}   

    bond = myPPP.bond


    Follower = Model('Follower_PPP')
    
    '''
    FOLLOWER VARIABLES
    '''
   
    x = {t:Follower.addVar(vtype=GRB.BINARY, name="x_"+str(t)) for t in range(myPPP.T)}                             # Whether a maintenance action is applied at t
    y = {t:Follower.addVar(vtype=GRB.INTEGER, name="y_"+str(t)) for t in range(myPPP.T)}					             # Number of periods after last restoration
    b = {(t,tau):Follower.addVar(vtype=GRB.BINARY, name="b_"+str((t,tau))) for t in range(myPPP.T) for tau in range(myPPP.T)}    # Whether yt=tau
    z = {(t,l):Follower.addVar(vtype=GRB.BINARY, name="z_"+str((t,l))) for t in range(myPPP.T) for l in myPPP.L}		      # Whether system is at service level l at t
    v = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="v_"+str(t)) for t in range(myPPP.T)}							# Performance at t
    pplus = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="earn_"+str(t)) for t in range(myPPP.T)}				# Earnings at t
    pminus = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="spend_"+str(t)) for t in range(myPPP.T)}				# Expenditures at t
    pdot = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="cash_"+str(t)) for t in range(myPPP.T)}				# Money at t
    w = {t:Follower.addVar(vtype=GRB.INTEGER, name="w_"+str(t)) for t in range(myPPP.T)}							# Linearization of y*x
    u = {t:Follower.addVar(vtype=GRB.CONTINUOUS, name="u_"+str(t)) for t in range(myPPP.T)}						# Lineartization for v*x
    aux = {(t,l):Follower.addVar(vtype=GRB.BINARY, name="aux_"+str((t,l))) for t in range(myPPP.T) for l in myPPP.L}              # variable for linearization ztl*qt
    Follower.update()
    '''
    OBJECTIVE
    '''
    #Follower Objective
    Follower.setObjective(-quicksum(pplus[t]-pminus[t] for t in range(myPPP.T)), GRB.MINIMIZE)
    '''
    FOLLOWER CONSTRAINTS
    '''
    #Initialization
    Follower.addConstr(y[0] == 0, "iniY") 
    Follower.addConstr(w[0] == 0, "iniW") 	
    Follower.addConstr(u[0] == 0, "iniU") 
    #Follower.addConstr(pdot[0] == pplus[0] - pminus[0], "cash_"+str(0)) 
    Follower.addConstr(pdot[0] == myPPP.S[2], "cash_"+str(0))
    
    for t in range(myPPP.T):
        if t>0:   
            # Restoration inventory
            Follower.addConstr(y[t] == y[t-1] + 1 - w[t] - x[t], "inv_"+str(t))
            
            # Linearization of w (for inventory)
            Follower.addConstr(w[t] <= y[t-1], "linW1_"+str(t))
            Follower.addConstr(w[t] >= y[t-1] - myPPP.T*(1-x[t]), "linW2_"+str(t))
            Follower.addConstr(w[t] <= myPPP.T*x[t], "linW3_"+str(t))
            
            # Linearization for v (to get ObjFcn right)
            Follower.addConstr(u[t] <= v[t], "linU1_"+str(t))
            Follower.addConstr(u[t] >= v[t] - (1-x[t]), "linU2_"+str(t))
            Follower.addConstr(u[t] <= x[t], "linU3_"+str(t))
            
            # Update available cash
            Follower.addConstr(pdot[t] == pdot[t-1] + pplus[t] - pminus[t], "cash_"+str(t))
            
        # Mandatory to improve the performance if it is below the minimum
        #HPR.addConstr(v[t] >= myPPP.minP, "minPerf_"+str(t))
        
        # Binarization of y (to retrieve performance)
        Follower.addConstr(y[t] == quicksum(tau*b[t,tau] for tau in range(myPPP.T)), "binY1_"+str(t))
        Follower.addConstr(quicksum(b[t,tau] for tau in range(myPPP.T)) == 1, "binY2_"+str(t))
        
        # Quantification of v (get performance)
        Follower.addConstr(v[t] == quicksum(gamma[tau]*b[t,tau] for tau in range(myPPP.T)), "quantV_"+str(t))
        
        # Linearization for service level
        Follower.addConstr(v[t] <= quicksum(xi_U[l]*z[t,l] for l in myPPP.L), "rangeU_"+str(t))
        Follower.addConstr(v[t] >= quicksum(xi_L[l]*z[t,l] for l in myPPP.L), "rangeL_"+str(t))
        
        # Specification of service-level (ranges)
        Follower.addConstr(quicksum(z[t,l] for l in myPPP.L) == 1, "1_serv_"+str(t))
        
        
        # Profit (budget balance)
        #HPR.addConstr(pplus[t] == myPPP.alpha + myPPP.f[t] + quicksum((myPPP.d[l,t+1]+myPPP.k[t])*z[t,l] for l in myPPP.L), "earn_"+str(t))
        Follower.addConstr(pminus[t] == (fc[t]+vc[t])*x[t]-vc[t]*u[t], "spend_"+str(t))
        Follower.addConstr(pminus[t] <= pdot[t] , "bud_"+str(t))
        
    # Return IMPORTANTEEEEEEEE
    #Follower.addConstr(quicksum(pplus[t] for t in range(myPPP.T)) >= (1+1.4*100)*quicksum(pminus[t] for t in range(myPPP.T)), "return")
    
    #Earnings quicksum(q[t]*z[t,l]*k[l] for l in myPPP.L) linealization
    for t in range(myPPP.T):
        for l in myPPP.L:
            Follower.addConstr(aux[t,l] <= x_param["q_"+str(t)], name = "binaux1_"+str((t,l)))
            Follower.addConstr(aux[t,l] <= z[t,l], name = "binaux2_"+str((t,l)))
            Follower.addConstr(aux[t,l] >= x_param["q_"+str(t)] + z[t,l] - 1, name = "binaux3_"+str((t,l)))
    
    for t in range(myPPP.T):
        Follower.addConstr(pplus[t] == quicksum(aux[t,l]*bond[l] for l in myPPP.L), name = "Agents_earnings_"+str(t)) #myPPP.a
    '''
    for t in myPPP.T:
        for l in myPPP.L:
            Follower.addConstr(x_param["aux_"+str((t,l))] <= z[t,l], name = "binaux2_"+str((t,l)))
            Follower.addConstr(x_param["aux_"+str((t,l))] >= x_param["q_"+str(t)] + z[t,l] - 1, name = "binaux3_"+str((t,l)))
    
    for t in myPPP.T:
        Follower.addConstr(pplus[t] == myPPP.a + quicksum(x_param["q_"+str(t)]*z[t,l]*myPPP.bond[l] for l in myPPP.L), name = "Agents_earnings_"+str(t))
    '''

    if maintenances:
        thyMaintenance = maintenances
        for name in thyMaintenance.keys():
             Follower.addConstr(Follower.getVarByName(name) == thyMaintenance[name])

    Follower.update() 

    return Follower

def comparison(policy, inspections, maintenances):
    # Status 2: feasible

    pareto = {"Principal": [], "Agent": []}

    gap_maint_agent = 0
    gap_maint_principal = 0
    gap_not_maint_principal = 0
    gap_not_maint_agent = 0
    it = 0
    for case in [maintenances, {}]:
        Follower = follower_PPP(inspections, case)
        Follower.optimize()
        if Follower.status == 2:
            it += 1
            social_benefit = sum(myPPP.g[l]*Follower.getVarByName("z_"+str((t,l))).x for l in myPPP.L for t in range(myPPP.T)) 
            cummulative_budget = -Follower.objVal

            if case:
                pareto["Principal"].append((social_benefit, "b", "envMaintenance"))
                pareto["Agent"].append((cummulative_budget, "b", "envMaintenance"))
                gap_maint_principal = social_benefit
                gap_maint_agent =   cummulative_budget
            else:
                pareto["Principal"].append((social_benefit, "r", "MIPMaintenance"))
                pareto["Agent"].append((cummulative_budget, "r", "MIPMaintenance"))
                gap_not_maint_principal = social_benefit
                gap_not_maint_agent =   cummulative_budget


    if it == 2:
        gap_principal = abs(gap_maint_principal - gap_not_maint_principal) / abs(gap_not_maint_principal)
        gap_agent = abs(gap_maint_agent - gap_not_maint_agent) / abs(gap_not_maint_agent)
    else:
        gap_principal = "No gap"
        gap_agent = "No gap"

    x_em = [pareto["Principal"][i][0] for i in range(len(pareto["Principal"])) if pareto["Principal"][i][1] == "b"]
    x_MIPm = [pareto["Principal"][i][0] for i in range(len(pareto["Principal"])) if pareto["Principal"][i][1] == "r"]
    y_em = [pareto["Agent"][i][0] for i in range(len(pareto["Agent"])) if pareto["Agent"][i][1] == "b"]
    y_MIPm = [pareto["Agent"][i][0] for i in range(len(pareto["Agent"])) if pareto["Agent"][i][1] == "r"]
    # col_ = [pareto["Principal"][i][1] for i in range(len(pareto["Principal"]))]
   
    #plt.scatter(x, y, c = col_, label = "Principal's vs. Agent Objective", s = 7)
    plt.scatter(x_em, y_em, c = "b", label = "envMaintenance", s = 7)
    plt.scatter(x_MIPm, y_MIPm, c = "r", label = "MIPMaintenance", s = 7)


    for x,y in zip(x_em,y_em):
            label = str((round(x,2),round(y,2)))
            plt.annotate(label, # this is the text
                        (x,y), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center

    for x,y in zip(x_MIPm,y_MIPm):
            label = str((round(x,2),round(y,2)))
            plt.annotate(label, # this is the text
                        (x,y), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center

    plt.legend(loc="best")
    plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
    plt.title("Principal's vs. Agent Objective_"+str(policy)+"_Gap: "+str(round(gap_agent,4)))
    plt.xlabel("Principal's objective")
    plt.ylabel("Agent's objective")
    save = True
    if save:
        plt.savefig("Principal's vs. Agent Objective_"+str(policy)+'.png')
    plt.show()
            


comparison("Fixed_5", dictInspections, dictMaints)
