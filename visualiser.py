import sys
import time
import numpy as np
import random
from player import Player
from env import Game
if sys.version_info.major == 2:
	import Tkinter as tk
else:
	import tkinter as tk



class Visualiser(tk.Tk, object):
	def __init__(self, env, scale):
		super(Visualiser, self).__init__()
		self.width = env.w
		self.height = env.h
		self.scale = scale

		self.title = "Bomber QL"
		self.geometry('{0}x{1}'.format(self.width*scale, self.height*scale))

		self.build_grid(env)

	def build_grid(self, env):
		self.canvas = tk.Canvas(self, bg='white',
						   height=self.height*self.scale,
						   width=self.width*self.scale)

		# make blocks
		for i in range(self.width):
			for j in range(self.height):
				if( i % 2 != 0 and j % 2 != 0 ):
					self.canvas.create_rectangle(
						i*self.scale, j*self.scale,
						i*self.scale + self.scale, j*self.scale + self.scale,
						fill='black')

		self.draw_players(env)

		self.canvas.pack()

	def update_canvas(self, env):
		time.sleep(0.2)
		self.canvas.delete(self.playerA)
		self.canvas.delete(self.playerB)

		self.draw_players(env)
		self.draw_bombs(env)
		self.update()

	def draw_players(self, env):
		# draw players
		self.playerA = self.canvas.create_rectangle(

						env.playerA.x*self.scale, env.playerA.y*self.scale,
						env.playerA.x*self.scale + self.scale, env.playerA.y*self.scale + self.scale,
						fill='red')
		# draw players
		self.playerB = self.canvas.create_rectangle(
						env.playerB.x*self.scale, env.playerB.y*self.scale,
						env.playerB.x*self.scale + self.scale, env.playerB.y*self.scale + self.scale,
						fill='blue')

	def draw_bombs(self, env):
		if(env.playerA.bomb_placed):
			print(env.playerA.bomb_life)
			if(env.playerA.bomb_life == 5):
				self.bombA = self.canvas.create_oval(
					env.playerA.bomb_pos[0]*self.scale, env.playerA.bomb_pos[1]*self.scale,
					env.playerA.bomb_pos[0]*self.scale + self.scale, env.playerA.bomb_pos[1]*self.scale + self.scale,
					fill='red')
			elif(env.playerA.bomb_life == 0):
				self.canvas.delete(self.bombA)

		if(env.playerB.bomb_placed):
			if(env.playerB.bomb_life == 5):
				self.bombB = self.canvas.create_oval(
					env.playerB.bomb_pos[0]*self.scale, env.playerB.bomb_pos[1]*self.scale,
					env.playerB.bomb_pos[0]*self.scale + self.scale, env.playerB.bomb_pos[1]*self.scale + self.scale,
					fill='blue')
			elif(env.playerB.bomb_life == 0):
				self.canvas.delete(self.bombB)



if __name__ == '__main__':
	env = Game()
	vis = Visualiser(env, 80)
	for i in range(100):
		env.step(random.randint(0,4),random.randint(0,4))
		vis.update_canvas(env)

	# env.step(4,1)
	# vis.update_canvas(env)
	# env.step(1,1)
	# vis.update_canvas(env)
	# env.step(1,1)
	# vis.update_canvas(env)
	# env.step(1,1)
	# vis.update_canvas(env)
	# env.step(1,1)
	# vis.update_canvas(env)
	# env.step(1,1)
	# vis.update_canvas(env)
	# env.step(1,1)
	# vis.update_canvas(env)
	# env.step(2,1)
	# vis.update_canvas(env)
	# env.step(2,1)
	# vis.update_canvas(env)
	# env.step(2,1)
	# vis.update_canvas(env)
