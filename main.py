from env import Game
import time
from visualiser import Visualiser
from randomplayer import *
from ql import QLearn

VIS = False
N_EPISODES = 5000

def test(cont=False, filename=None):
	numActions = env.n_actions
	playerA = QLearn(actions=list(range(numActions)), reward_decay=0.7)
	if(cont):
		playerA.load_Qtable(filename)
		playerA.save_Qtable("old_actions")
	playerB = RandomPlayer(numActions-1)
	start_time = time.time()
	for episode in range(N_EPISODES):
		# initial observation
		observation = env.reset()
		# print(str(episode))
		if(episode % 500 == 0):
			print(str(float(episode) / N_EPISODES * 100) + "%")
		while True:
			# RL choose action based on observation
			actionA = playerA.choose_action(str(observation))
			actionB = playerB.choose_action(str(observation))

			# RL take action and get next observation and reward
			observation_, reward, done = env.step(actionA, actionB)
			#print(observation_)

			# RL learn from this transition
			playerA.learn(str(observation), actionA, reward, str(observation_), done)
			# swap observation
			observation = observation_

			# break while loop when end of this episode
			if done:
				break
	# end of game
	print "My program took", time.time() - start_time, "to run"
	print('game over')
	playerA.save_Qtable("actions")
	print(playerA.check_convergence("old_actions"))

def testB(cont=False, filename=None):
	numActions = env.n_actions
	playerA = QLearn(actions=list(range(numActions)))
	if(cont):
		playerA.load_Qtable(filename)
	playerB = QLearn(actions=list(range(numActions)))
	start_time = time.time()
	for episode in range(N_EPISODES):
		# initial observation
		observation = env.reset()
		# print(str(episode))
		if(episode % 500 == 0):
			print(str(float(episode) / N_EPISODES * 100) + "%")
		while True:
			# RL choose action based on observation
			actionA = playerA.choose_action(str(observation))
			actionB = playerB.choose_action(str(observation))

			# RL take action and get next observation and reward
			observation_, reward, done = env.step(actionA, actionB)
			#print(observation_)

			# RL learn from this transition
			playerA.learn(str(observation), actionA, reward, str(observation_), done)
			playerB.learn(str(observation), actionB, -reward, str(observation_), done)
			# swap observation
			observation = observation_

			# break while loop when end of this episode
			if done:
				break
	# end of game
	print "My program took", time.time() - start_time, "to run"
	print('game over')
	playerA.save_Qtable("actions")
	playerB.save_Qtable("actionsB")


def run_optimal():
	vis = Visualiser(env, 80)
	numActions = env.n_actions
	playerA = QLearn(actions=list(range(numActions)), e_greedy=1.0)
	playerA.load_Qtable("actions")
	playerB = RandomPlayer(numActions-1)
	for episode in range(20):
		observation = env.reset()
		vis.update_canvas(env)
		while(True):
			actionA = playerA.choose_action(str(observation))
			actionB = playerB.choose_action(str(observation))
			observation_, reward, done = env.step(actionA, actionB)
			observation = observation_
			vis.update_canvas(env)
			if done:
				vis.reset()
				break
	print("Games won: " + str(env.win_count))
	vis.destroy()

def run_optimalB():
	vis = Visualiser(env, 80)
	numActions = env.n_actions
	playerA = QLearn(actions=list(range(numActions)), e_greedy=1.0)
	playerA.load_Qtable("actions")
	playerB = QLearn(actions=list(range(numActions)), e_greedy=1.0)
	playerB.load_Qtable("actionsB")
	for episode in range(20):
		observation = env.reset()
		vis.update_canvas(env)
		while(True):
			actionA = playerA.choose_action(str(observation))
			actionB = playerB.choose_action(str(observation))
			observation_, reward, done = env.step(actionA, actionB)
			observation = observation_
			vis.update_canvas(env)
			if done:
				vis.reset()
				break
	print("Games won: " + str(env.win_count))
	vis.destroy()

if __name__ == '__main__':
	env = Game()
	# testB(cont=True, filename="actions")
	# test(cont=True, filename="actions")
	run_optimal()
	# run_optimalB()