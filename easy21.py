import numpy as np
from tqdm import tqdm
import pickle as pk
from classes import *

def generate_episode(env, policy, control_method, Q_table = None, Count_table = None):
    '''
    policy is a function object which takes state as input
    '''
    env.reset()
    state_ls = []; action_ls = []; reward_ls = []
    while env.state.is_terminal != True:
        # observable state
        observe_state = (env.state.agent_sum, env.state.dealer_first_card_value)
        if control_method == 'GLIE Monte Carlo Control':
            # Follow GLIE monte carlo control
            action = policy(observe_state, Q_table, Count_table)
        
        state_ls.append(observe_state)
        action_ls.append(action)
        env.state, reward = env.step(env.state, action)
        reward_ls.append(reward)
        episode = (state_ls, action_ls, reward_ls)
    return episode

def update_action_value_estimate(episode, Q_table, Count_table):
    state_ls, action_ls, reward_ls = episode
    g_return = 0
    gamma = 1
    for _ in range(len(state_ls)):
        # Go backward to update return
        state = state_ls.pop(); action = action_ls.pop(); reward = reward_ls.pop()
        g_return = gamma * g_return + reward

        # Map state to state_idx, action to action_idx
        i = Count_table.state_to_idx[state]; j = Count_table.action_to_idx[action]
        
        # Update N(s,a)
        Count_table.table[i][j] += 1
        
        # Update q(s,a) estimate
        error_signal = g_return - Q_table.table[i][j]
        Q_table.table[i][j] += (1/Count_table.table[i][j]) * error_signal
    return Q_table

def GLIE_mc_control(num_episode=0, env=None, agent=None):
    '''
    GLIE every-vist Monte Carlo Control
    '''
    Q_table = lookup_tabular(env.state_space, env.action_space)
    Count_table = lookup_tabular(env.state_space, env.action_space)
    control_method = 'GLIE Monte Carlo Control'

    for _ in tqdm(range(num_episode), desc=control_method):
        # On-policy
        policy = agent.policy
        episode = generate_episode(env, policy, control_method, Q_table = Q_table, Count_table=Count_table)
        Q_table = update_action_value_estimate(episode, Q_table, Count_table)
    
    return Q_table

# init env and agent
env = easy21()
agent = eps_soft_agent()

# Monte_carlo
Q_table = GLIE_mc_control(num_episode=1000000, env=env, agent=agent)

with open('Q_table.pk', 'wb') as f:
    pk.dump(Q_table, f)
    
del Q_table