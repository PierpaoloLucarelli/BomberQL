import numpy as np
from player import Player

# must be odd
WIDTH = 7
HEIGHT = 7
STEP_SIZE = 2


class Game():

	def __init__(self):
		self.w = WIDTH
		self.h = HEIGHT
		self.actions = ['u','r','d','l','b']
		self.n_actions = len(self.actions)
		self.grid = np.array((WIDTH,HEIGHT))
		self.playerA = Player((0,0))
		self.playerB = Player((WIDTH-1, HEIGHT-1))

	def reset(self):
		self.playerA = Player((0,0))
		self.playerB = Player(WIDTH-1, HEIGHT-1)
		# return a board to state



	def step(self, actionA, actionB):
		self.apply_action(actionA, self.playerA)
		self.apply_action(actionB, self.playerB)


	def apply_action(self, action, player):
		base_action = np.array([0,0])
		if action == 0: # up
			if(player.y > 0):
				base_action[1] -= 1
		elif action == 1: #right
			if player.x < WIDTH-1:
				base_action[0] += 1
		elif action == 2: #down
			if player.y < HEIGHT-1:
				base_action[1] += 1
		elif(action == 3): # left
			if(player.x > 0):
				base_action[0] -= 1
		else:
			print("action not valid")
		if (player.x + base_action[0]) % 2 != 0 and (player.y + base_action[1]) % 2 != 0:
			return
		player.x += base_action[0]
		player.y += base_action[1]




