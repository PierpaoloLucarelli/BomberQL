from env import Game
import time
from randomplayer import *
from ql import QLearn

VIS = False
N_EPISODES = 5000

def test():
	numActions = env.n_actions
	playerA = QLearn(actions=list(range(numActions)))
	playerB = RandomPlayer(numActions)
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
	RL.save_Qtable()

if __name__ == '__main__':
	env = Game()
	test()