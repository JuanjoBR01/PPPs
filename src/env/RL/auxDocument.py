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
# -------------------------------- lIBRARY IMPORT ---------------------------------

from math import exp, log, sin
import matplotlib.pyplot as plt
import numpy as np
import random
import degradationProof as dp

plt.style.use('ggplot')
from gurobipy import *
np.random.seed(10)
random.seed(10)

SHOW_EVERY = 1000

# -------------------------------- CLASS DEFINITION ---------------------------------

class EnvPPP():

    # -------------------- Builder Method --------------------

    def __init__(self):

        # ---- Class Atributes ----
        self.S = [0, 1, 2]                          # Current state (for when running [Periods Without Maintenance, Performance, Budget]
        self.T = 30								    # Planning horizon
        self.W = [0, 1] 		                    # [shock?, inspection?] Binary parameters
        self.NUM_INTERVALS = 5
        self.L = range(1, self.NUM_INTERVALS + 1)   # Discrete levels of performance

        self.FC = 1                                 # Fixed maintenance cost
        self.VC = 1                                 # Variable maintenance cost
        self.rate = 3 								# "Slope" of sigmoidal benefit-performance function
        self.offset = 3 							# "Offset" of sigmoidal benefit-performance function
        self.threshold = 0.6                        # Performance treshold (discrete)

        # ---- Hyperparameters -----

        self.LEARNING_RATE = 0.1
        self.DISCOUNT = 0.9
        self.epsilon = 0.15
        self.START_EPSILON_DECAYING = 1             # Episode where epsilon begins to affect
        self.END_EPSILON_DECAYING = self.T // 2     # Episode where epsilon does not affect anymore

        # Rate of decay of epsilon after each step
        self.epsilon_decay_value = self.epsilon / (self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING)

        self.fail = .2										# failure threshold (continuous)
        self.ttf = 10  										# time to failure

        # perf_tt = exp(-Lambda*tt) --> Lambda = -ln(perf_tt)/tt
        self.Lambda = -log(self.fail)/self.ttf

        # ---- Q-Table ----
        # Dimensions: ttf x decisions(2)
        # The environment will decide according to the days without maintenance in the system
        self.q_table = np.random.uniform(low=-35, high=-34.5, size=([self.ttf] + [2]))


        self.f_W = [(1, [False, False])]                    # scenarios: (prob_s, [shock, inspection])
        self.gamma = .9										# discount factor
        self.v_hat = {i: 0 for i in range(int(self.ttf))}   # state value estimation

        self.alpha = .5

        # Social benefit related to the performance level. g_star is the expected one
        self.g = {1: 2, 2: 47, 3: 500, 4: 953, 5: 998}
        # Earnings target
        self.g_star = 595

        # Deterioration function that depends in the periods without maintenance
        self.gamma = [exp(-self.Lambda*tau) for tau in range(self.T+2)]

        # Bond establishes the incentive that depends in the discrete performance level
        self.bond = {}
        for level in self.L:
            average_l = 0
            count_l = 0
            for gamma_val in self.gamma:
                if self.get_level(gamma_val) == level:
                    average_l += (7)*self.incentive(gamma_val)
                    count_l += 1
            self.bond[level] = average_l/count_l

        # Inspection policy
        self.inspection_policy = 'fixed_5'


    # Initial conditions in period 0
    def reset(self):
        return [0, 1, 2]

    # Discretization of the performance levels
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
        return self.get_level(perf) - 1

    def incentive(self, perf, choice='sigmoid'):
        if self.W[1] == 0:
            return 0

        if choice == 'sigmoid':
            rate, offset = 10, self.threshold
            incent = 1/(1 + exp(-rate*(perf-offset)))

        elif choice == 'linear':
            slope, offset = 1, 0
            incent = offset + slope*perf

        return incent

    # Incentive that depends in the inspection (binary parameter) 
    def MIP_incentive(self):
        if self.W[1] == 0:
            return 0

        return self.bond[self.get_level(self.S[1])]

    # Function to decide when will the government inspect the project
    def inspect(self, policy, episode):
        # Three options: fixed (every 5 periods), random_x (bernoulli probability), reach (if the performance reaches a determined level, will inspect)
        if policy[:5] == 'fixed':
            aux = episode + 1
            return 1 if episode > 0 and aux % int(policy[6:]) == 0 else 0

        elif policy[:6] == 'random':
            aux = random.uniform(0,100)
            return 1 if aux < int(policy[7:]) else 0

    # Cost function depending on the incentive and the maintenance decision
    def cost(self, X):
        return -self.FC*X - self.VC*X + self.MIP_incentive()

    # Transition between states function
    def transition(self, X):
        cf = self.cost(X)
        if X:
            self.S = [0, 1, self.S[2]+cf]
            self.gamma = dp.generate(30,1,'P',(dp.s_shape,1,30,0.1))[0]
        else:
            self.S[0] += 1
            self.S[1] = self.gamma[self.S[0]+1]
            self.S[2] += cf
        return cf

    # Function that decides whether an inspection is done or not
    def fixed_action_rule_agent(self, random_exploration):
        # The q_value is the number of periods without performance
        discrete_state = self.S[0]

        # If there's one period left until failure, the maintenance is mandatory
        if discrete_state == self.ttf-1:
            return 1
        # If there's no budget available, it won't fix
        elif self.S[2] < self.FC + self.VC:
            return 0

        # If both feasibility conditions passed, random exploration is reviewed
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
        plt.savefig('Performance' + self.inspection_policy +'.png')
        plt.show()

    # Function to iterate the environment
    def run(self, state, episode_rewards_agent, aggr_ep_rewards_agent, show_it):
        # performance = [state[1]]
        # cashflow = [state[2]]
        performance = []
        cashflow = []
        inspections = []
        maintenances = []
        cashflow = []
        budget = []
        X = 0

        for episode in range(self.T):
            # Decide if a random exploration is done
            if np.random.random() > self.epsilon:
                X = self.fixed_action_rule_agent(False)
            else:
                X = self.fixed_action_rule_agent(True)
            self.W[1] = self.inspect(self.inspection_policy, episode)
            inspections.append(self.W[1])
            maintenances.append(X)
            # Update of the q_table
            prev_state = self.S[0]
            current_q = self.q_table[prev_state, X]
            reward = self.transition(X)
            new_state = self.S[0]
            max_future_q = np.max(self.q_table[new_state])
            new_q = (1-self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q)
            self.q_table[prev_state, X] = new_q

            '''
            episode_rewards_agent.append(reward)

            if not show_it % SHOW_EVERY:    
                average_reward_agent = sum(episode_rewards_agent[-SHOW_EVERY:])/len(episode_rewards_agent[-SHOW_EVERY:])
                aggr_ep_rewards_agent['ep'].append(show_it)
                aggr_ep_rewards_agent['avg'].append(average_reward_agent)
                aggr_ep_rewards_agent['min'].append(min(episode_rewards_agent[-SHOW_EVERY:]))
                aggr_ep_rewards_agent['max'].append(max(episode_rewards_agent[-SHOW_EVERY:]))

                save = True

                if show_it % SHOW_EVERY == 0 and show_it > 0 and save:
                    moving_avg_agent = np.convolve(episode_rewards_agent, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode= "valid")

                    fig, ax1 = plt.subplots()
                    #ax1 = ax1.twinx()
                    ax1.plot([j for j in range(len(moving_avg_agent))], moving_avg_agent, color = 'b', label = "Agents's performance", linestyle = 'dashed')
                    ax1.set_ylabel(f"Agent moving avg reward every {SHOW_EVERY} episodes")
                    ax1.legend(loc='upper right')
                    plt.suptitle("Learning performance", fontsize=15)
                    plt.grid(True)
                    plt.savefig('Learning_performance.png')
                    plt.close()

            show_it += 1
            '''

            if episode == 0:
                performance.append(self.S[1])
                budget.append(self.S[2])
                cashflow.append(round(reward, 5))
            else:
                performance.append(self.S[1])
                budget.append(self.S[2])
                cashflow.append(round(reward, 5))

        return budget, inspections, maintenances, performance, cashflow, show_it


def follower_PPP(x_param, maintenances):
    
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
    Follower.setParam(GRB.Param.OutputFlag, 0)
    #Follower.setParam(GRB.Param.InfUnbdInfo, 1)
    #Follower.setParam(GRB.Param.DualReductions, 0)


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


        elif Follower.status == 3:
                status = "Infeasible"
                print(f'\nThe optimization status is: {status} (code {Follower.status})')
	
        else:
            status = "Unbounded or something else"
            print(f'\nThe optimization status is: {status} (code {Follower.status})')


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

    results = policy + ',' + str(x_em[0]) + ',' + str(y_em[0]) + ',' + str(x_MIPm[0]) + ',' + str(y_MIPm[0]) + ',' +str(gap_agent)

    return results


def show_pareto(route):

    results = []

    with open(route) as f:
        while True:
            line = f.readline()

            if not line:
                break

            values = line.split(',')
            elements = {'policy': values[0], 'x_env': float(values[1]), 'y_env': float(values[2]), 'x_MIP': float(values[3]), 'y_MIP': float(values[4]), 'gap': float(values[5])}
            results.append(elements)
        
        policies = []
        x_env = []
        y_env = []
        x_MIP = []
        y_MIP = []

        for i in range(len(results)):
            policies.append(results[i]['policy'])
            x_env.append(results[i]['x_env'])
            y_env.append(results[i]['y_env'])
            x_MIP.append(results[i]['x_MIP'])
            y_MIP.append(results[i]['y_MIP'])

        # Env data plot
        # Fixing random state for reproducibility

        N = len(policies)
        colors = np.random.rand(N)
        
        plt.scatter(x_env, y_env, c= 'b', alpha=0.5, label = "Agents Maintenance")
        plt.scatter(x_MIP, y_MIP, c= 'r', alpha=0.5, label = "Public Maintenance")
        plt.title("Principal vs Agent's Objective")
        plt.xlabel("Principal's objective")
        plt.ylabel("Agent's objective")
        plt.show()




                
# ----------------------- DECLARING, INITIALIZING AND RUNNING THE ENVIRONMENT -----------------------

# Policies to evaluate:
policy = 'random_25'
#policies = [ 'fixed_1', 'fixed_2', 'fixed_3', 'fixed_5']
            #'random_80', 'random_60', 'random_50', 'random_25']

# Declaration of the instance
myPPP = EnvPPP()
#myPPP.gamma = dp.generate(31,1,'P',(dp.s_shape,1,30,0.1))
#myPPP.gamma = dp.generate(30,1,'P',(dp.s_shape,1,30,0.1))[0]

'''
fig, ax = plt.subplots()
ax.plot(range(32), myPPP.gamma)
plt.title("Deterioro Determinístico con Función Sigmoidal")
plt.ylabel("Nivel de desempeño")
plt.xlabel("Periodo")

plt.show()
'''


fig, ax = plt.subplots()
for i in range(30):
    ax.plot(range(30), dp.generate(30,1,'P',(dp.s_shape,1,30,0.1))[0])
plt.title("Deterioro Estocástico con Procesos Gamma")
plt.ylabel("Performance")
plt.xlabel("Period")
plt.show()



