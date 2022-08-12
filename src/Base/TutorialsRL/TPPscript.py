"""
@author: juanbeta
"""

### Basic Librarires
import numpy as np; from copy import copy, deepcopy; import matplotlib.pyplot as plt
import networkx as nx; import sys

### Gym & OR-Gym
import gym; from gym import spaces; from or_gym import utils

### Supporting Modules
sys.path.insert(1, '/Users/juanbeta/Dropbox/9 Semestre/Tesis/OR - Gym/TPP/Supporting Modules')
from Generate_Locations import generate_locations; from CrearDemanda import imprimir_disponibilidad
# from CrearDistancias import matriz_distancias

### GRASP MIP Algorithm
sys.path.insert(1, '/Users/juanbeta/Dropbox/9 Semestre/Tesis/OR - Gym/TPP/GRASP Algorithm')
from TPP_CODEv3 import Corre_Modelo_Matematico



'''
#######################################  Definitons  ####################################### 

## 1.1 State
The state is defined as a vector with dimensions 1 + N + M. The first component is an integer 
indiacting the current node. The following N components describe for every store if it has 
already been, or cannot be visited (0) or if it has not been visited yet and at least one of 
its commodities hasn't been purchased. Finally, the last M components indicates for every 
commodity if it has been (0) or not (1) purchased yet.

[Entero,   N,    M]

## 1.2 Action
The action is defined as a vector with 1 + M components. The first component indicates the 
next node to be visited. The following M components indicate which of the commodities (1) 
will be purchased in the next choosen node.

## 2 Instance
The instance is generated using a deterministic approach by Isabella (2020).
The method takes the seed, number of products, number of nodes and a disponibility scale:
- Disponibility: 1 = High; 2 = Medium; 3 = Low

## 3 Generation of actions
For Q-learning algorithms, actions are generated depending on a given approach. For both 
approaches, the number of actions generated are the minimum between 5 and the number of 
commodities that are not purchased yet. The aproaches are:
- Greedy: For the feasible chosen nodes, all of the feasible commodities will be chosen
          to purchase at each node
- K - number: For the feasible chosen nodes, the maximum between K and the available 
          commodities in tat node will be chosen to purchase at each node

'''


### Class TPPenv
class TPPenv(gym.Env):
    
    # Initializing the environment
    def __init__(self, file_name = '', *args, **kwargs):
        
        ### Parameters
        self.N = 4                                              # Number of nodes 
        self.M = 4                                              # Number of commodities
        self.invalid_move_cost = 1000                                # Cost of choosing invalid arch 
        self.dispo = 2                                          # Availability of the products
        self.seed = 1                                           # Seed for random numbers generation
        self.p_min = 20                                         # Minimum price
        self.p_max = 30                                         # Maximum price
        self.alpha = 0.5
        
        utils.assign_env_config(self, kwargs)                   # Apply custom configurations
        
        self.upload = False
        if file_name != '':
            self.N, self.M, self.demand, self.comm_dict,\
                self.pric_dict, self.dist_dict = self.upload_instance(file_name)
            self.upload = True
        
        self.nodes = np.arange(self.N)                          # List of nodes 
        self.markets = list(range(1,self.N))                    # List of markets
        self.commodities = np.arange(self.M)                    # List of commodities
        
        ### State space
        self.obs_dim = self.N + self.M                      # See 1.1
        state_low = np.zeros(self.obs_dim)               
        state_max = np.hstack([self.N, np.repeat(1, self.N + self.M - 1)])
        self.observation_space = spaces.Box(state_low, state_max, dtype = np.int32)
        
        ### Action space
        act_dim = 1 + self.M                                    # See 1.2
        self.act_dim = act_dim
        action_low = np.hstack([1, np.repeat(0, act_dim - 1)])                     # Mininmum value = 0              
        action_high = np.hstack([self.N - 1, 
                                 np.repeat(1, self.M)])         # Maximun value = [N, 1]
        self.action_space = spaces.Box(action_low, action_high, dtype = np.int32)    
        
        ### Instance generation                                 # See 2
        if  not self.upload:
            self.comm_dict, self.pric_dict = \
                imprimir_disponibilidad(self.seed, self.M, 
                                        self.N, self.dispo, 
                                        self.p_min, self.p_max)     # Generate dispnibilty and prices
            self.dist_dict,  self.coord_dict = generate_locations(self.N, self.seed)
        self.min_price = {i: np.min([self.pric_dict[j][i] for j in self.markets if self.pric_dict[j][i] != 0]) \
                          for i in self.commodities}            # Minimum price for every commodity
    
    # Reseting the environment
    def reset(self):
        
        self.current_node = 0                                   # Starting at depot (0)
        self.route = []
        self.generate_dictionaries()
        self.state = self.assemble_state()
        
        return self.state                                       # Return state
    
    # Generate the dictionaries containing environment's information
    def generate_dictionaries(self):
        
        self.visit_log = {i:1 for i in self.markets}            # Visited nodes log
        self.purchase_log = {i:0 for i in self.commodities}     # Purchased commodities log
        self.not_v_nodes = copy(self.markets)                   # List of not visited nodes
        if not self.upload:
            self.demand = {i: 1 for i in self.commodities}      # Demand of commodities
        self.offer = {i: [j for j in self.markets if \
                          self.comm_dict[j][i] == 1] for i in self.commodities}     # Offering markets for comm
             
    # Give a step when a given action is performed  
    def step(self, action):
                                                                       
        valid, target, reward = self.check_validity(action)
        done = False
        
        if valid:                                               # Update state 
            reward = self.update_state(action, target) 
            done, reward = self.check_termination(reward)

        return self.state, reward, done, {}                     # Return state, eward and termination condition
    
    # Check validity of the action
    def check_validity(self, action):
        
        valid = True
        target = action[0] 
        print(f'target: {target}'); print(f'action: {action}')
        reward = 0
        
        if self.visit_log[target] == 0:                         # Check if already visit
            reward = self.invalid_move_cost
            valid = False     
        elif sum(action[1:]) == 0:                              # Check if at least one commodity is purchased
            reward = self.invalid_move_cost
            valid = False   
        else:
            for i in self.commodities:
                com = action[i + 1]
                if  com > self.comm_dict[target][i] or \
                    com == 1 and self.purchase_log[i] == 1:     # Check if choosen commodities are 
                    valid = False                               # offered or already purchased
                    reward = self.invalid_move_cost
                    break
        
        return valid, target, reward
    
    # Check episode's termination
    def check_termination(self, reward):
        
        done = False 
        
        if sum([self.purchase_log[i] for i in self.commodities]) >= self.M:     # Al items are purchased
            reward += self.dist_dict[self.current_node][0]
            self.route.append((self.current_node,0))
            done = True
        
        elif sum([self.visit_log[i] for i in self.markets]) == 0:               # All markets are visited        
            reward += self.invalid_move_cost
            done = True
        
        else:
            not_purchased_items = [i for i in self.commodities if self.purchase_log[i] == 0]    
            if True not in [i in self.offer[j] for j in not_purchased_items for i in self.not_v_nodes]:
                reward += self.invalid_move_cost                # At least one more item can be purchased
                done = True
        
        return done, reward
    
    # Update dictionaries, route and state
    def update_state(self, action, target):
        self.not_v_nodes.remove(target)
        start = self.current_node
        self.current_node = target                              # Update current node 
        self.route.append((start,target))                               
        self.visit_log[self.current_node] = 0                   # Update visit log
        pur = [max(self.purchase_log[i], action[i + 1])         # Update purchase log
             for i in self.commodities]                 
        self.purchase_log = {i: pur[i] for i in self.commodities}      
        self.state = self.assemble_state()                       
        reward = np.dot(self.pric_dict[self.current_node], action[1:]) + \
            self.dist_dict[start][self.current_node]                          # Reward = price + transportaton
        
        return reward
    
    # Assemble the state's vector
    def assemble_state(self):
        
        return np.hstack([self.current_node,
                        list(self.visit_log.values()),
                        list(self.purchase_log.values())
                        ])  

    # Render the environment 
    def render(self):
        G2 = nx.DiGraph(self.route)
        G2.add_nodes_from(self.nodes)
        options = {
        "font_size": 10,
        "node_size": 300,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1
        }
        fig, ax = plt.subplots()
        nx.draw_networkx(G2, self.coord_dict, **options)
        # Set margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis('on')
        ax.tick_params(left = True, bottom = True, labelleft = True, labelbottom = True)
        plt.xlabel('Longitud') 
        plt.ylabel('Latitud')
        plt.title('Ruta Elegida') 
        plt.show()
    
    # Gereate actions for Q-learning (Greedy and k-items)
    def generate_actions(self, heur_type):                  # See 3
        alternatives = []                                   # Empty alternative list
        falt = self.M - sum(self.state[self.N:])            # ??? Items left to purchase
        num = min(5, falt)                                  # ??? Number of alternatives
        pos = deepcopy(self.not_v_nodes)                    # pos = nodes that have not been visited yet
        for i in range(0, num):
            alt = []                                        # Current alternative
            done = False                                    # Not filled yet
            if pos != []:
                while not done:
                    
                    target = self.select_node(pos)          # Random not visited node
                    pos.remove(target)                      # Remove node from list of not visited nodes
                    if pos == []:                           # Check if no market offers commodities left
                        done = True
                    if True in list((self.purchase_log[i], self.comm_dict[target][i]) == (0,1)\
                                        for i in self.commodities): # If there is a not purchased item in that store
                        if heur_type == 'greedy':
                            
                            alt.append(target)
                            alt += list(int((self.purchase_log[i], self.comm_dict[target][i]) == (0,1))\
                                        for i in self.commodities)
                            alternatives.append(alt)
                            done = True            
                        else:   
                            heur_type_n = self.M * heur_type
                            alt.append(target)
                            if sum(list(int((self.purchase_log[i], self.comm_dict[target][i]) == (0,1))\
                                        for i in self.commodities)) <= heur_type_n:
                               
                                alt += list(int((self.purchase_log[i], self.comm_dict[target][i]) == (0,1))\
                                        for i in self.commodities)
                                alternatives.append(alt)
                                done = True
                            else:  
                                alt1 = list(int((self.purchase_log[i], self.comm_dict[target][i]) == (0,1))\
                                        for i in self.commodities)
                                done1 = False
                                comm1 = list(self.commodities)   
                                while not done1: 
                                    chosen = np.random.choice(comm1)
                                    comm1.remove(chosen)
                                    alt1[chosen] = 0
                                    if sum(list(int((self.purchase_log[i], alt1[i]) == (0,1))\
                                        for i in self.commodities)) <= heur_type_n:  
                                        alt += alt1
                                        alternatives.append(alt)
                                        done1 = True  
                                        done = True 
        return alternatives
    
    # Auxiliary function to generate_actions
    def select_node(self,pos):
        
        dist_tot = 0
        for i in pos:
            dist_tot += self.dist_dict[self.current_node][i]
        
        probabilities = [self.dist_dict[self.current_node][i] / dist_tot for i in pos]
        threshold = np.random.random()
        
        for j in pos:
            if sum(probabilities[:j]) > threshold:
                target = j
                break
            
        return target
      
    # Generate actions for Q-learning (RCL and probability choosing based on price)
    def generate_actions2(self):
        
        alternative = []
        RCL = self.generate_RCL() 
        options = list(np.where(RCL)[0])
        
        target = np.random.choice(options) + 1
        alternative.append(target)
        posib = []
        
        for i in self.commodities:
            price = self.pric_dict[target][i]
            if (self.purchase_log[i], self.comm_dict[target][i]) == (0,1):
                posib.append(i)
                if price == self.min_price[i]:
                    alternative.append(1)
                else:
                    prob_comp = 0.5 * (1 - (price - self.min_price[i])/price)
                    if np.random.random() < prob_comp:
                        alternative.append(1)
                    else:
                        alternative.append(0)
            else:
                alternative.append(0)
        
        # Check if at least one item is purchased
        if sum(alternative[1:]) == 0:
            number = min([self.pric_dict[target][i] for i in posib])
            alternative = [target]
            for i in self.commodities:
                if self.pric_dict[target][i] == number:
                    alternative.append(1)
                else:
                    alternative.append(0)
        
        return alternative
    
    # Generate RCL
    def generate_RCL(self):
        
        list_inter = [] 
    
        for mark in self.markets: 
            if self.visit_log[mark] == 0:
                list_inter.append(0)
            else:
                exist = False
                for comm in self.commodities:
                    if self.purchase_log[comm] == 0 and mark in self.offer[comm]:
                        exist = True 
                list_inter.append(self.dist_dict[self.current_node][mark] * exist)
        
        max_dist = np.max(list_inter)
        min_dist = np.min([i for i in list_inter if i > 0])
        limit = min_dist + self.alpha * (max_dist - min_dist)
        RCL = list((int(0 < i <= limit) for i in list_inter))
        
        return RCL
    
    # Solution of the Mixed Integer Program
    def MIP_solution(self, objective):
        
        V = self.nodes
        M = self.markets
        K = self.commodities
        
        Cij = {}
        for i in self.nodes:
            for j in self.nodes:
                if self.dist_dict[i][j] != 0 and i != j:
                    Cij[(i,j)] = self.dist_dict[i][j]
                    
        fwd = {l:[(i,j) for (i,j) in Cij.keys() if i==l] for l in V}
        rev = {l:[(i,j) for (i,j) in Cij.keys() if j==l] for l in V}
        
        fik = {}
        qik = {}
        for i in self.markets:
            for j in self.commodities:
                qik[(i,j)] = self.comm_dict[i][j]
                fik[(i,j)] = self.pric_dict[i][j]
        
        Mk = {}
        for prod in self.commodities: 
            lista = []
            for i in range(1,self.N):
                if self.comm_dict[i][prod] == 1:
                    lista.append(i)
            Mk[prod] = lista
        
        dk = {k:1 for k in self.commodities}
        
        Corre_Modelo_Matematico(V, M, K, fwd, rev, Cij, fik, qik, Mk, dk, objective)
    
    # Upload an exixiting instance in .tpp format
    def upload_instance(self, file_name):
        
        file = open(file_name, mode = 'r')   
        file = file.readlines()
        
        N = int(file[3][-4:])
        M = int(file[10])
        
        demand = {}
        for i in range(0,M):
            
            # TODO: Needs to be adjusted in case demand is > 10
            demand[i] = int(file[i + 11][-3:])
        
        
        comm_dict = {}
        pric_dict = {}
        for i in range(0, N-1):      
            
            row = str(file[M + i + 13]).split()
            dict1 = {}
            dict2 = {}
            for j in range(0, int((len(row) - 2)/3)):
                commodity = int(row[2 + j * 3])
                for h in range(0, M):
                    if h + 1 == commodity:
                        dict1[commodity] = 1
                        dict2[commodity] = int(row[3 + j * 3])
            
            comm_vector = [1 * (l + 1 in dict1.keys()) for l in range(0,M)]
            pric_vector = []
            for y in range(0, M):
                if comm_vector[y] != 0:
                    pric_vector.append(dict2[y + 1])
                else:
                    pric_vector.append(0)
            
            comm_dict[i+1] = comm_vector
            pric_dict[i+1] = pric_vector
            
            
        dist_dict = {}
        for node_i in range(0,N):
            
            row_list = []
            row = str(file[M + node_i + 13 + N]).split()
            
            for node_f in range(0,N):
        
                row_list.append(int(row[node_f]))
             
            dist_dict[node_i] = row_list
        
        return N, M, demand, comm_dict, pric_dict, dist_dict

    # Generate file with the instance
    def generate_file(self):
        sys.stdout = open("/Users/juanbeta/Dropbox/9 Semestre/Tesis/OR - Gym/TPP/Inst" + "." + str(self.dispo) + "." + \
                          str(self.N) + "." + str(self.M) + "." + str(self.seed) +".tpp", "w")
        
        #Fill in the file
        print('NAME \t :') 
        print('TYPE \t : TPP')
        print('COMMENT \t :') 
        print("Markets: " + str(self.N))
        print("Products: " + str(self.M))
        print("Availability: " + str(self.dispo))
        print("Seed: " + str(self.seed))
        print('DEMAND_SECTION:')
        print(f'{self.M}')
        for i in self.commodities:
            print(f'{i} {self.demand[i]}')
        print("OFFER_SECTION:")
        print('1 0')
        for i in range(1, self.N):
            print(f'{i} {sum(self.comm_dict[i])}  ' + self.print_dispos(i))
        print("DISTANCE_SECTION:")
        for i in self.nodes:
            print(self.print_dist(i))
        print("EOF")

    # Auxiliary function (generate_file)
    def print_dispos(self, Node):
        string = ''
        for i in self.commodities:
            if self.comm_dict[Node][i] == 1:
                string += f'{i} {self.pric_dict[Node][i]} 1 '
        return string

    # Auxiliary function (generate_file)
    def print_dist(self, Node):
        string = ''
        for i in self.nodes: 
            string += '\t' + f'{self.dist_dict[Node][i]}' + '\t'
        return(string)

