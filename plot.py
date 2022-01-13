
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
import pickle as pk
import numpy as np
from classes import *


def find_optimal_value_action(Q_table):
    dealer_show_ls = np.array([i for i in range(1,11)])
    agent_sum_ls = np.array([j for j in range(1,22)])
    optimal_action_matrix = np.zeros((len(agent_sum_ls), len(dealer_show_ls)))
    optimal_value_matrix = np.zeros((len(agent_sum_ls), len(dealer_show_ls)))

    agent_ls = []; dealer_ls = []; optimal_value_ls = []; optimal_action_ls = []
    for player_sum in agent_sum_ls:
        for dealer_show in dealer_show_ls:
            state = (player_sum, dealer_show)
            state_idx = Q_table.state_to_idx[state]
            optimal_value = max(Q_table.table[state_idx,:])
            optimal_action = np.argmax(Q_table.table[state_idx,:])

            agent_ls.append(player_sum)
            dealer_ls.append(dealer_show)
            optimal_value_ls.append(optimal_value)
            optimal_action_ls.append(optimal_action)
            optimal_action_matrix[player_sum-1][dealer_show-1] = optimal_action
            optimal_value_matrix[player_sum-1][dealer_show-1] = optimal_value
    return optimal_value_matrix, optimal_action_matrix, (agent_ls, dealer_ls, optimal_value_ls)

def plot_value_function(value_space):
    #Plot value function
    agent_ls, dealer_ls, optimal_value_ls = value_space
            
    fig = plt.figure()
    ax = plt.axes(projection ='3d') 
    cmap = plt.get_cmap('hot')
    trisurf = ax.plot_trisurf(agent_ls, dealer_ls, optimal_value_ls, linewidth=0.2, cmap=cmap)
    fig.colorbar(trisurf, ax = ax, shrink = 0.5, aspect = 10)
    ax.view_init(20, 210)
    ax.set_xlabel('player sum',fontweight ='bold')
    ax.set_ylabel('dealer face card',fontweight ='bold')
    ax.set_zlabel('value',fontweight ='bold')
    ax.set_xlim(1,21)
    ax.set_xticks(np.arange(1,22,2))
    ax.set_title('Optimal Value Function V*(s) ')
    plt.draw()

def plot_policy_map(optimal_action_matrix):
    # create discrete policy map
    fig = plt.figure()
    ax = plt.axes()
    cmap = colors.ListedColormap(['red', 'blue'])
    ax.imshow(optimal_action_matrix, cmap = cmap)
    
    #draw gridlines
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.grid(True, which='minor', axis='both', linestyle='-', color='k', linewidth=2)
    
    # Fixing x axis
    x_tick = np.arange(0, 10, 1)
    ax.set_xlim(-0.5,9.5)
    x_ticklabel = [i+1 for i in x_tick]
    ax.set_xticks(x_tick)
    ax.set_xticklabels(x_ticklabel)
    
    # Fixing y axis
    y_tick = np.arange(0,21,1)
    ax.set_ylim(-0.5, 20.5)
    y_ticklabel = [i+1 for i in y_tick]
    ax.set_yticks(y_tick)
    ax.set_yticklabels(y_ticklabel)
    
    # Label
    ax.set_xlabel('dealer face card',fontweight ='bold')
    ax.set_ylabel('player sum',fontweight ='bold')
    ax.text(10,10, 'red = hit\nblue = stick')
    #fig.legend(['hit', 'stick'])
    plt.show()


with open('Q_table.pk', 'rb') as f:
    Q_table = pk.load(f)


optimal_value_matrix, optimal_action_matrix, value_space = find_optimal_value_action(Q_table)
plot_value_function(value_space)
plot_policy_map(optimal_action_matrix)

