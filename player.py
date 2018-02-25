import time 

BOMB_LIFE = 8 # time steps

class Player:
	def __init__(self, pos):
		self.x = pos[0]
		self.y = pos[1]
		self.bomb_placed = False
		self.bomb_life = BOMB_LIFE
		self.bomb_pos = None

	def place_bomb(self, pos):
		# pos is a tuple (x,y)
		self.bomb_placed = True
		self.bomb_pos = pos

	def has_bomb(self):
		return not self.bomb_placed

	def reset_bomb(self):
		self.bomb_placed = False
		self.bomb_life = BOMB_LIFE
		self.bomb_pos = None

	def __str__(self):
		return str(self.x) + ", " + str(self.y)