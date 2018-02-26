import numpy as np


class RandomPlayer:

    def __init__(self, numActions):
        self.numActions = numActions

    def choose_action(self, state, restrict=None):
        return np.random.randint(self.numActions)

    def getReward(self, initialState, finalState, actions, reward, maxActions=None):
        pass