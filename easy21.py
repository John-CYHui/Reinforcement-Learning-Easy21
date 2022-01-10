import numpy as np

class State():
    def __init__(self) -> None:
        self.dealer_sum = 0
        self.agent_sum = 0
        self.is_terminal  = False
        pass
    
class environment(State):
    def __init__(self) -> None:
        super().__init__()
        card_value, card_color = self.initial_draw()
        self.dealer_sum = self.update_sum(card_value, card_color, self.dealer_sum)
        card_value, card_color = self.initial_draw()
        self.agent_sum = self.update_sum(card_value, card_color, self.agent_sum)
    
    def initial_draw(self):
        card_value = np.random.randint(low=1,high=10)
        # Initial draw must be black
        card_color = 1
        return card_value, card_color
    
    def draw(self):
        card_value = np.random.randint(low=1,high=10)
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
                self.agent_sum = self.update_sum(card_value, card_color, self.agent_sum)
                state.agent_sum = self.agent_sum
                if self.is_bust(self.agent_sum):
                    reward = -1
                    state.is_terminal = True
            elif action == 'stick':
                while self.dealer_sum < 17:
                    card_value, card_color = self.draw()
                    self.dealer_sum = self.update_sum(card_value, card_color, self.dealer_sum)
                    state.dealer_sum = self.dealer_sum
                    if self.is_bust(self.dealer_sum):
                        reward = 1
                        state.is_terminal = True
                        break
                if self.dealer_sum > self.agent_sum:
                    reward = -1
                    state.is_terminal = True
                elif self.dealer_sum == self.agent_sum:
                    reward = 0
                    state.is_terminal = True
                if self.dealer_sum < self.agent_sum:
                    reward = 1
                    state.is_terminal = True
        return state, reward
# init
env = environment()
state = State()
print(state.agent_sum)

for i in range(10):
    state, reward = env.step(state, 'stick')
    print(state.agent_sum, state.dealer_sum, state.is_terminal, reward)