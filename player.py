import time 

class Player:
	def __init__(self, pos):
		self.x = pos[0]
		self.y = pos[1]
		self.bomb_placed = False

	def place_bomb(self, pos):
		# pos is a tuple (x,y)
		self.bomb_placed = True
		self.bomb_pos = pos
		self.bomb_time = int(round(time.time() * 1000))

	def __str__(self):
		return str(self.x) + ", " + str(self.y)