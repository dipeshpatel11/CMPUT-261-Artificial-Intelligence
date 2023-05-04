import numpy as np

from action_value_table import ActionValueTable

# HINT: You will need to sample from probability distributions to complete this
# question.  Please use `numpy.random.rand` to generate a floating point number
# uniformly at random and `numpy.random.choice` to uniformly randomly select an
# element from a list.
#
# Documentation:
# - https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
# - https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3  # agents actions
GAMMA = 0.95
STEP_SIZE = 0.25
EPSILON = 0.1


class QLearningAgent():
    """
    Implement your code for a Q-learning agent here. We have provided code
    implementing the action-value table in `action_value_table.py`. Here, you
    will implement the `get_action`, `get_greedy_action` and `update` methods.
    """

    def __init__(self, dimension):
        self.actions = [UP, DOWN, LEFT, RIGHT]
        num_actions = len(self.actions)
        self.values = ActionValueTable(dimension, num_actions)
        self.gamma = GAMMA
        self.step_size = STEP_SIZE
        self.epsilon = EPSILON


    def update(self, state, action, reward, next_state, done):
        """
        Update the values stored in `self.values` using Q-learning.

        HINT: Use `self.values.get_value` and `self.values.set_value`
        HINT: Remember to add a special case to handle the terminal state

        Args:
            state : (list) a list of type [bool, int, int] where the first
            entry is whether the agent posseses the key, and the next two
            entries are the row and column position of the agent in the maze
            
            action : (int) the action taken at state

            reward : float

        Returns:
            None
        """
        #Q(S,A)<- Q(S,A)+ alpha(R+gamma.maxa Q(S',a)Q(S,A)
        #S<-S'
        #If the episode has ended i.e done, the value of the next state is 0 and update the value of the action that led to the terminal state
        if done:
            update_value = self.step_size * (reward + (self.gamma * 0) - self.values.get_value(state, action)) + self.values.get_value(state, action)
            
            self.values.set_value(state, action, update_value)
        # Otherwise, estimate the value of the next state using the maximum Q-value
        else:
            actions_list = []
            
            for a in self.actions:
                actions_list.append(self.values.get_value(next_state, a))
            # maximum action-value for the next state
            update_value = self.step_size * (reward + (self.gamma * np.max(actions_list)) - self.values.get_value(state, action)) + self.values.get_value(state, action) 
            
            self.values.set_value(state, action, update_value)


    def get_action(self, state):
        """
        This function returns an action from self.actions given a state. 

        Implement this function using an epsilon-greedy policy. 

        HINT: use np.random.rand() to generate random numbers
        HINT: If more than one action has maximum value, treat them all as the
        greedy action. In other words, if there are b greedy actions, each
        should have probability
        (1 - epsilon)/b + epsilon/|A|,
        where |A| is the number of actions in this state.

        Args:
            state : (list)
            a list of type [bool, int, int] where the first entry is whether
            the agent posseses the key, and the next two entries are the row
            and column position of the agent in the maze

        Returns:
            action : (int) a epsilon-greedy action for `state`
        """
        prob_list = []
        #If random number is less than epsilon, then the agent selects a random action from available actions
        if np.random.rand() < self.epsilon:      
            return np.random.choice(self.actions)
        
        # Otherwise, it checks which action has the maximum value for the given state. 
        else:
            max_value_list = [] #keep track of the actions that have the maximum value for a given state.
            max_value = 0
            
            for action in self.actions: 
                if (self.values.get_value(state, action) > max_value):
                    new_max_list = []
                    max_value = self.values.get_value(state, action)
                    new_max_list.append(action)
                    max_value_list = new_max_list
                elif (self.values.get_value(state, action) == max_value):
                    max_value_list.append(action)

            if len(max_value_list) == 1:  # If there is only one action with maximum value, choose it
                return max_value_list[0]
            # Otherwise, it calculates the probability of selecting each action with the maximum value
            else:
                probability = ((1 - self.epsilon) / len(max_value_list)) + (self.epsilon/abs(len(max_value_list)))
                for i in range(len(max_value_list)):
                    prob_list.append(probability)
                    
                 # selects one of them randomly based on these probabilities.
                return np.random.choice(max_value_list, p = prob_list)


    def get_greedy_action(self, state):
        """
        This function returns an action from self.actions given a state. 

        Implement this function using a greedy policy, i.e. return the action
        with the highest value
        
        HINT: If more than more than one action has maximum value, uniformly
        randomize amongst them

        Args:
            state : (list)
            a list of type [bool, int, int] where the first entry is whether
            the agent posseses the key, and the next two entries are the row
            and column position of the agent in the maze

        Returns:
            action : (int) a greedy action for `state`
        """
        
        max_value_list = []
        max_value = 0
        
        for action in self.actions:
            if (self.values.get_value(state, action) > max_value):
                max_value = self.values.get_value(state, action)
                new_max_list = []
                new_max_list.append(action)
                max_value_list = new_max_list
            elif (self.values.get_value(state, action) == max_value):
                max_value_list.append(action)
                
                
        action_choice = np.random.choice(max_value_list)

        return action_choice                      