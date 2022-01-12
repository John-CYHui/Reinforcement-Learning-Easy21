import numpy as np
class lookup_tabular():
    def __init__(self, state_space, action_space):
        self.table = np.zeros((len(state_space),len(action_space)))
        self.init_state_dict(state_space)
        self.init_idx_lookup_dict(state_space, action_space)

    def init_state_dict(self, state_space):
        self.state_count_dict = {}
        for state in state_space:
            # N(s)
            self.state_count_dict[state] = 0

    def init_idx_lookup_dict(self, state_space, action_space):
        self.state_to_idx = {}
        self.action_to_idx = {}; self.idx_to_action = {}
        for idx, state in enumerate(state_space):
            self.state_to_idx[state] = idx
        for idx, action in enumerate(action_space):
            self.action_to_idx[action] = idx
            self.idx_to_action[idx] = action


class easy21():
    class State():
        def __init__(self) -> None:
            self.dealer_sum = 0
            self.agent_sum = 0
            self.dealer_first_card_value = None
            self.is_terminal  = False

    def __init__(self) -> None:
        self.state = self.State()
        self._state_space()
        self._action_space()
        self.reset()
    
    def _state_space(self):
        self.state_space = tuple((player_sum, dealer_first_card_value) for player_sum in range(1,22) for dealer_first_card_value in range(1,11))
    
    def _action_space(self):
        self.action_space = ['hit', 'stick']    
        
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


class eps_soft_agent:
    def __init__(self) -> None:
        self.action = None

    def get_epsilon(self, observe_state, Count_table):
        N_0 = 100
        i = Count_table.state_to_idx[observe_state]
        N_s = np.sum(Count_table.table[i, :])
        epsilon = N_0 / (N_0 + N_s)
        return epsilon

    def greedy_action(self, observe_state, Q_table):
        i = Q_table.state_to_idx[observe_state]
        # Find the row in Q value table corresponding to state S_t 
        q_s_a = Q_table.table[i,:]
        # Find the greedy idx that gives maximum Q value
        j = np.argmax(q_s_a)
        # Map greedy idx back to greedy action
        greedy_action = Q_table.idx_to_action[j]
        return greedy_action

    def policy(self, observe_state, Q_table, Count_table):
        '''Follow epsilon-soft policy, given a state from env, return an action'''
        # greedy action
        greedy_action = self.greedy_action(observe_state, Q_table)
        # epsilon scheduling
        epsilon = self.get_epsilon(observe_state, Count_table)
        self.epsilon = epsilon
        # return a epsilon greedy action
        rand_num = np.random.uniform(0,1)
        if rand_num > epsilon:
            eps_greedy_action = greedy_action
        else:
            rand_choice = np.random.choice([0,1])
            eps_greedy_action = Q_table.idx_to_action[rand_choice]
        return eps_greedy_action

