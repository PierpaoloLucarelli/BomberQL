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
		self.update_bomb(self.playerA)
		self.update_bomb(self.playerB)
		self.apply_action(actionA, self.playerA)
		self.apply_action(actionB, self.playerB)
		if(self.playerB.bomb_placed or self.playerA.bomb_placed):
			self.check_death(self.playerA)
			self.check_death(self.playerB)



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
		elif(action == 4): # place bomb
			if(player.has_bomb()):
				player.place_bomb((player.x, player.y))
		else:
			print("action not valid")
		if (player.x + base_action[0]) % 2 != 0 and (player.y + base_action[1]) % 2 != 0:
			return
		player.x += base_action[0]
		player.y += base_action[1]

	def update_bomb(self, player):
		if(player.bomb_placed):
			if(player.bomb_life > 0):
				player.bomb_life -= 1
			else:
				print("explosion")
				player.reset_bomb()


	def check_death(self, player):
		if(self.playerA.bomb_placed):
			if(player.x == self.playerA.bomb_pos[0]):
				if(abs(player.x - self.playerA.bomb_pos[0]) < 2):
					return True
			if(player.y == self.playerA.bomb_pos[1]):
				if(abs(player.y - self.playerA.bomb_pos[1]) < 2):
					return True
		if(self.playerB.bomb_placed):
			if(player.x == self.playerB.bomb_pos[0]):
				if(abs(player.x - self.playerB.bomb_pos[0]) < 2):
					return True
			if(player.y == self.playerB.bomb_pos[1]):
				if(abs(player.y - self.playerB.bomb_pos[1]) < 2):
					return True
		
		return False






