# %%
#%%
import numpy as np
import random
from tqdm import tqdm

# %%
class environment():
    class State():
        def __init__(self) -> None:
            self.dealer_sum = 0
            self.agent_sum = 0
            self.dealer_first_card_value = None
            self.is_terminal  = False

    def __init__(self) -> None:
        self.state = self.State()
        self.reset()
    
    def reset(self):
        self.state.dealer_sum = 0
        self.state.agent_sum = 0
        card_value, card_color = self.draw(black_only=True)
        self.state.dealer_sum = self.update_sum(card_value, card_color, self.state.dealer_sum)
        self.state.dealer_first_card_value = self.state.dealer_sum
        card_value, card_color = self.draw(black_only=True)
        self.state.agent_sum = self.update_sum(card_value, card_color, self.state.agent_sum)
        self.state.is_terminal  = False

    def draw(self, black_only = False):
        card_value = np.random.randint(low=1,high=11)

        if black_only == True:
            card_color = 1
            return card_value, card_color
        else:
            rand_num = np.random.uniform(low=0, high=1)
            # Define red = 0, black = 1
            if rand_num < 1/3:
                card_color = 0
            else:
                card_color = 1
            return card_value, card_color
    
    def update_sum(self, card_value, card_color, accum_sum):
        if card_color == 0:
            accum_sum -= card_value
        elif card_color == 1:
            accum_sum += card_value
        return accum_sum
    
    def is_bust(self, accum_sum):
        if accum_sum > 21 or accum_sum < 1:
            return True
        else:
            return False 

    def step(self, state, action):
        reward = 0
        if state.is_terminal != True:
            if action == 'hit':
                card_value, card_color = self.draw()
                state.agent_sum = self.update_sum(card_value, card_color, state.agent_sum)
                if self.is_bust(state.agent_sum):
                    reward = -1
                    state.is_terminal = True
            elif action == 'stick':
                while state.dealer_sum < 17 and state.dealer_sum >= 1:
                    card_value, card_color = self.draw()
                    state.dealer_sum = self.update_sum(card_value, card_color, state.dealer_sum)

                if self.is_bust(state.dealer_sum):
                    reward = 1
                    state.is_terminal = True
                elif state.dealer_sum > state.agent_sum:
                    reward = -1
                    state.is_terminal = True
                elif state.dealer_sum == state.agent_sum:
                    reward = 0
                    state.is_terminal = True
                elif state.dealer_sum < state.agent_sum:
                    reward = 1
                    state.is_terminal = True
        return state, reward



# %%
class Agent:
    def __init__(self) -> None:
        self.action = None
        self.state_space = tuple((player_sum, dealer_first_card_value) for player_sum in range(1,22) for dealer_first_card_value in range(1,11))
        self.action_space = ['hit', 'stick']
        self.state_action_space = tuple((player_sum, dealer_first_card_value, action) for player_sum in range(1,22) for dealer_first_card_value in range(1,11) for action in self.action_space)
        self.init_state_dict()
        self.init_idx_lookup_dict()
        self.init_state_action_lookup_table()

    def init_state_dict(self):
        self.state_count_dict = {}
        for state in self.state_space:
            # N(s)
            self.state_count_dict[state] = 0

    def init_idx_lookup_dict(self):
        self.state_to_idx = {}
        self.action_to_idx = {}; self.idx_to_action = {}
        for idx, state in enumerate(self.state_space):
            self.state_to_idx[state] = idx
        for idx, action in enumerate(self.action_space):
            self.action_to_idx[action] = idx
            self.idx_to_action[idx] = action

    def init_state_action_lookup_table(self):
        self.action_value_lookup_table = np.zeros((len(self.state_space),len(self.action_space)))
        self.state_action_count_lookup_table = np.zeros((len(self.state_space),len(self.action_space)))

    def get_epsilon(self, observe_state):
        N_0 = 100
        N_s = self.state_count_dict[observe_state]
        epsilon = N_0 / (N_0 + N_s)
        # Update count
        N_s += 1
        self.state_count_dict[observe_state] = N_s
        return epsilon

    def greedy_action(self, observe_state):
        i = self.state_to_idx[observe_state]
        # Find the row in Q value table corresponding to state S_t 
        q_s_a = self.action_value_lookup_table[i,:]
        # Find the greedy idx that gives maximum Q value
        j = np.argmax(q_s_a)
        # Map greedy idx back to greedy action
        greedy_action = self.idx_to_action[j]
        return greedy_action

    def eps_soft_policy(self, observe_state):
        '''Follow epsilon-soft policy, given a state from env, return an action'''
        # greedy action
        greedy_action = self.greedy_action(observe_state)
        # epsilon scheduling
        epsilon = self.get_epsilon(observe_state)
        self.epsilon = epsilon
        # return a epsilon greedy action
        eps_greedy_action = np.random.choice([greedy_action, np.random.choice(self.action_space)], p = [1-epsilon, epsilon])
        return eps_greedy_action


# %%
def generate_episode(env, policy):
    '''
    policy is a function object which takes state as input
    '''
    env.reset()
    state_ls = []; action_ls = []; reward_ls = []
    while env.state.is_terminal != True:
        # observable state
        observe_state = (env.state.agent_sum, env.state.dealer_first_card_value)
        # Follow policy pi
        action = policy(observe_state)
    
        state_ls.append(observe_state)
        action_ls.append(action)
        env.state, reward = env.step(env.state, action)
        reward_ls.append(reward)
        #print(env.state.agent_sum, env.state.dealer_sum, env.state.is_terminal, reward)
        episode = (state_ls, action_ls, reward_ls)
    return episode

def update_action_value_estimate(agent, episode):
    state_ls, action_ls, reward_ls = episode
    g_return = 0
    gamma = 1
    for _ in range(len(state_ls)):
        # Go backward to update return
        state = state_ls.pop(); action = action_ls.pop(); reward = reward_ls.pop()
        g_return = gamma * g_return + reward

        # Map state to state_idx, action to action_idx
        i = agent.state_to_idx[state]; j = agent.action_to_idx[action]
        
        # Update N(s,a)
        agent.state_action_count_lookup_table[i][j] += 1
        
        # Update q(s,a) estimate
        error_signal = g_return - agent.action_value_lookup_table[i][j]
        agent.action_value_lookup_table[i][j] += (1/agent.state_action_count_lookup_table[i][j]) * error_signal
    return agent.action_value_lookup_table

def GLIE_mc_control(num_episode=0, env=None, agent=None):
    '''
    GLIE every-vist Monte Carlo Control
    '''
    for _ in tqdm(range(num_episode)):
        policy = agent.eps_soft_policy
        episode = generate_episode(env, policy)
        agent.action_value_lookup_table = update_action_value_estimate(agent, episode)
# %%
# init
env = environment()
agent = Agent()

# Monte_carlo
GLIE_mc_control(num_episode=500000, env=env, agent=agent)


#%%
# Plot value function
dealer_show_ls = np.array([i for i in range(1,11)])
agent_sum_ls = np.array([j for j in range(1,22)])

optimal_action_matrix = np.zeros((len(agent_sum_ls), len(dealer_show_ls)))
optimal_value_matrix = np.zeros((len(agent_sum_ls), len(dealer_show_ls)))

agent_ls = []; dealer_ls = []; optimal_value_ls = []; optimal_action_ls = []
for player_sum in agent_sum_ls:
    for dealer_show in dealer_show_ls:
        state = (player_sum, dealer_show)
        state_idx = agent.state_to_idx[state]
        optimal_value = max(agent.action_value_lookup_table[state_idx,:])
        optimal_action = np.argmax(agent.action_value_lookup_table[state_idx,:])

        agent_ls.append(player_sum)
        dealer_ls.append(dealer_show)
        optimal_value_ls.append(optimal_value)
        optimal_action_ls.append(optimal_action)
        optimal_action_matrix[player_sum-1][dealer_show-1] = optimal_action
        optimal_value_matrix[player_sum-1][dealer_show-1] = optimal_value
        
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(agent_ls, dealer_ls, optimal_value_ls, linewidth=0.2, cmap=plt.cm.viridis)
ax.view_init(20, 210)
plt.draw()

# %%
# Policy debug
test_state = (3, 10)
action = agent.eps_soft_policy(test_state)
print(agent.epsilon, action)


# %%
test = agent.action_value_lookup_table
print(agent.state_to_idx[test_state])

# %%
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

data = np.random.rand(10, 10) * 20

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


