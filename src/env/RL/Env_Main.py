import PPP_Env_V2

# ----------------------- DECLARING, INITIALIZING AND RUNNING THE ENVIRONMENT -----------------------

# Declaration of the instance
myPPP = PPP_Env_V2.EnvPPP()

# Declaration of the cumulative q_table, in the next replica, the agent will take the q_table that the previous agent left
new_q_table =  myPPP.q_table

# Number of simulations to be run. The self.T will be ran the times in the parameter
num_simulations = int(1e4)

# Cumulative rewards in order to know how the agent is learning
rewards = []

# Number of periods of each. It depends on the class atribute

for i in range (num_simulations):
    state = myPPP.reset()
   # myPPP.epsilon = thyEpsilon
    myPPP.S = state

    if i == num_simulations-1:
        myPPP.epsilon = 0


    myPPP.q_table = new_q_table
    values = []
    values = myPPP.run(state)

    new_q_table = myPPP.q_table

    rewards.extend(values[4])

    ''' 
    if myPPP.epsilon > 0.15:
        myPPP.epsilon *= 0.998
    else: 
        myPPP.epsilon = 0.15

    if myPPP.epsilon <= 0.15 and myPPP.epsilon > 0.08:
        myPPP.epsilon *= 0.5
    elif myPPP.epsilon < 0.08: 
        myPPP.epsilon = 0.15
    thyEpsilon = myPPP.epsilon
    '''
    

# The order of the values array is [cashflow(0), inspections(1), maintenances(2), performance(3), reward(4)]
myPPP.show_performance(values[1], values[2], values[3])
#myPPP.show_cashflow(values[0])
#myPPP.show_rewards(rewards, num_simulations * myPPP.T)


# Section of the code that prints the decisions made by the agent at the end of the simulation among 30 periods

dictMaints = {}
for i in range(len(values[2])):
    dictMaints['x_' + str(i)] = values[2][i]

print(dictMaints)