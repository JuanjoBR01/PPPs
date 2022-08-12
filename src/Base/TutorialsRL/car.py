'''
---------------------- Q Learning Tutorial - Sentdex ----------------------
------------------------- Juan José Beltrán Ruiz --------------------------
'''

# Source code: https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/
# Specific problem: https://gym.openai.com/envs/MountainCar-v0/ 

# Actions:
# 0: Push the car left
# 1: Do nothing
# 2: Push the car right

# State is composed by position and velocity

# Library import
import gym
import numpy as np
import matplotlib.pyplot as plt

# Declaration and initialization of the environment

env = gym.make("MountainCar-v0")
env.reset()


# Declaration of hyperparameters 

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 500




''' 
print(env.observation_space.high)                                                                           # Max discrete state
print(env.observation_space.low)                                                                            # Min discrete state
print(env.action_space.n)                                                                                   # Number of possible actions
''' 

# Discretization parameters

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)                                                   # Discretize the states in order don't spend all the memory in small decimal precision
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE          # Length of the discrete intervals


# Random exploration parameters

epsilon = 0.5                                                                                               # Probability to make a random exploration
START_EPSILON_DECAYING = 1                                                                                  # Episode where epsilon begins to affect
END_EPSILON_DECAYING = EPISODES // 2                                                                        # Episode where epsilon does not affect anymore
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)                               # Rate of decay of epsilon after each step


# Declaration of the q table and lists to collect information

q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))           # Each state and action has the Q value 
ep_rewards = []                                                                                             # Rewards of each episode
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max':[]}                                                # Aggregate measures along the episodes


# Function to get the discrete state and print test commented

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


'''
print(discrete_state)
print(np.argmax(q_table[discrete_state]))
'''

for episode in range(EPISODES):                                                                             # Loop for all the learning episodes
    
    episode_reward = 0

    if episode % SHOW_EVERY == 0:                                                                           # Render the graph according to the hyperparameter
        print (episode)
        render = True
    else:
        render = False


    discrete_state = get_discrete_state(env.reset()) 
    print(discrete_state)                                                       # Each episode begins from the initial position
    done = False

    while not done:
        
        if np.random.random() > epsilon:                                                                    # Decide if a random exploration is done
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)
        
        new_state, reward, done, _ = env.step(action)                                                       # New values according to the transition function

        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)                                                  # Get the new state

        if render:
            env.render()

        if not done:                                                                                        # If the car has not arrived to the flag, then continue
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)      # Bellman equation from https://wikimedia.org/api/rest_v1/media/math/render/svg/678cb558a9d59c33ef4810c9618baf34a9577686

            q_table[discrete_state + (action, )] = new_q                                                    # Update Q value in the Q table

        elif new_state[0]  >= env.goal_position:                                                            # If the car gets to the flag, then don't penalize in the Q table
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state                                                                 # Discretize the new state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:                                           # Review if the epsilon must be decayed
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)                                                                       # Add the episode reward

    if not episode % SHOW_EVERY:                                                                            # Get the aggregate functions
        #np.save(f"qtablescar\{episode}-qtable.npy", q_table)
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode}, avg: {average_reward}, min: {min(ep_rewards[-SHOW_EVERY:])}, max: {max(ep_rewards[-SHOW_EVERY:])}")


env.close()                                                                                                 # Close the connection to the environment

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = 'avg')                                      # Plot :)
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = 'min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = 'max')

plt.legend(loc = 4)
plt.show()














