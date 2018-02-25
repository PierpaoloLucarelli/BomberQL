class Player:
	def __init__(self, pos):
		self.x = pos[0]
		self.y = pos[1]

	def __str__(self):
		return str(self.x) + ", " + str(self.y)