
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle as pk
import numpy as np
from classes import *

with open('Q_table.pk', 'rb') as f:
    Q_table = pk.load(f)

# Plot value function
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
        

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(agent_ls, dealer_ls, optimal_value_ls, linewidth=0.2, cmap=plt.cm.viridis)
ax.view_init(20, 210)
plt.draw()


# create discrete colormap
cmap = colors.ListedColormap(['red', 'blue'])
bounds = [0,0.5,1]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
ax.imshow(optimal_action_matrix, cmap=cmap, norm=norm)

# draw gridlines
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
ax.set_xticks(np.arange(1, 10, 1))
ax.set_yticks(np.arange(1, 21, 1))
#ax.legend(['hit', 'stick'])
plt.show()

