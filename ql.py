import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import cPickle as pickle

class QLearn:
	def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
		self.actions = actions  # a list
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

	# chose the next action to take based on the observation (position of actor)
	def choose_action(self, observation, available=None):
		self.check_state_exist(observation)
		# print(self.q_table)
		# based on epsilon, chose either the best action or a random action (eploration vs exploitation)
		if np.random.uniform() < self.epsilon:
			# chose best action epsilon % of the time
			# based on the 4 coordinates of the actor
			state_action = self.q_table.ix[observation, :]
			if(available):
				for i in range(len(state_action)):
					if( i not in available ):
						state_action.drop(i, inplace=True)
			# this returns a labbelled array where label is action and val is Qval of the action
			state_action = state_action.reindex(np.random.permutation(state_action.index)) # make sure to not pick always the first value
			# pick action with max Qval
			action = state_action.idxmax()
		else:
			#pick a random val
			if(available):
				action = np.random.choice(available)
			else:
				action = np.random.choice(self.actions)
		return action

	def learn(self, s, a, r, s_, done, available=None):
		self.check_state_exist(s_)
		# get the Q value of the action a at state s
		q_predict = self.q_table.ix[s, a]
		# print(q_predict)
		
		if done == False:
			state_action = self.q_table.ix[s_, :]
			# print(state_action)
			if(available):
				for i in range(len(state_action)):
					if( i not in available ):
						state_action.drop(i, inplace=True)
			# print(state_action)
			q_target = r + self.gamma * state_action.max()  # next state is not terminal
		else:
			q_target = r  # next state is terminal
		self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

	def set_greedy(self, val):
		self.epsilon = val


	def check_state_exist(self, state):
		if state not in self.q_table.index:
			# append new state to q table
			self.q_table = self.q_table.append(
				pd.Series(
					[0]*len(self.actions),
					index=self.q_table.columns,
					name=state,
				)
			)

	def check_convergence(self, filename):
		try:
			data = pd.read_pickle(filename)
			return assert_frame_equal(data, self.q_table, check_dtype=False)
		except AssertionError:
			print("Not Converged")
			print("number of old states: " + str(len(data)))
			print("number of new states: " + str(len(self.q_table)))
			return False


	def save_Qtable(self, filename):
		self.q_table.to_pickle(filename)

	def load_Qtable(self, filename):
		self.q_table = pd.read_pickle(filename)