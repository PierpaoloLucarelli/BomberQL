from env import Game
import time
from visualiser import Visualiser
from randomplayer import *
from ql import QLearn
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from minimax_ql import MinimaxQPlayer
from dqn import DeepQNetwork

VIS = False
N_EPISODES = 1000

# test QL vs Random
def test(cont=False, filename=None):

	reward_a = np.zeros(N_EPISODES)
	total_a = 0

	numActions = env.n_actions
	playerA = QLearn(actions=list(range(numActions)), reward_decay=0.7)
	playerB = QLearn(actions=list(range(numActions)), reward_decay=0.7)
	if(cont):
		playerA.load_Qtable(filename)
		playerA.save_Qtable("old_actions")
	playerB.load_Qtable('actions_copy')
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
			total_a += reward

			# RL learn from this transition
			playerA.learn(str(observation), actionA, reward, str(observation_), done)
			# swap observation
			observation = observation_

			# break while loop when end of this episode
			if done:
				break
		reward_a[episode] = total_a
	# plt.plot(reward_a)
	# plt.ylabel('Cummulative reward')
	# plt.xlabel('Episode')
	# plt.show()
	# playerA.save_Qtable("qlactions")
	return playerA

# test minmaxQL vs Random
def test_minmax(cont=False, filename=None):

	reward_a = np.zeros(N_EPISODES)
	total_a = 0

	numActions = env.n_actions
	drawProbability = 0.1
	decay = 10**(-2. / N_EPISODES * 0.05)

	playerA = MinimaxQPlayer(numActions, numActions, decay=decay, expl=0.01, gamma=1-drawProbability)
	if(cont):
		playerA.load_Qtable(filename)
		playerA.save_Qtable("old_actions")
	playerB = QLearn(actions=list(range(numActions)), reward_decay=0.7)
	playerB.load_Qtable("qlactions")
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
			total_a += reward

			# RL learn from this transition
			playerA.learn(str(observation), str(observation_), [actionA,actionB], reward)
			# swap observation
			observation = observation_

			# break while loop when end of this episode
			if done:
				break
		reward_a[episode] = total_a
	# plt.plot(reward_a)
	# plt.ylabel('Cummulative reward')
	# plt.xlabel('Episode')
	# plt.show()
	return playerA


	# end of game
	print "My program took", time.time() - start_time, "to run"
	print('game over')
	playerA.save_Qtable("actions")
	print(playerA.check_convergence("old_actions"))

def testB(cont=False, filenames=None):
	numActions = env.n_actions
	playerA = QLearn(actions=list(range(numActions)), reward_decay=0.7)
	playerB = QLearn(actions=list(range(numActions)), reward_decay=0.7)
	if(cont):
		playerA.load_Qtable(filenames[0])
		playerA.save_Qtable("old_actions")
		playerB.load_Qtable(filenames[1])
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
			actionB = playerB.choose_action(str(observation), list(range(numActions)))

			# RL take action and get next observation and reward
			observation_, reward, done = env.step(actionA, actionB)
			#print(observation_)

			# RL learn from this transition
			playerA.learn(str(observation), actionA, reward, str(observation_), done)
			# playerB.learn(str(observation), actionB, -reward, str(observation_), done)
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
	print(playerA.check_convergence("old_actions"))


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

def test_DQL():
	reward_a = np.zeros(N_EPISODES)
	total_a = 0

	step = 0
	numActions = env.n_actions
	playerA = DeepQNetwork(numActions, env.n_features,
					  learning_rate=0.01,
					  reward_decay=0.9,
					  e_greedy=0.9,
					  replace_target_iter=200,
					  memory_size=2000,
					  # output_graph=True
					  )
	playerB = RandomPlayer(numActions-1)
	for episode in range(N_EPISODES):
		# initial observation
		observation = env.reset()
		if(episode % 500 == 0):
			print(str(float(episode) / N_EPISODES * 100) + "%")
		while True:
			# RL choose action based on observation
			# print(observation)
			actionA = playerA.choose_action(np.array(observation))
			actionB = playerB.choose_action(str(observation))
			# RL take action and get next observation and reward
			observation_, reward, done = env.step(actionA, actionB)
			total_a += reward
			playerA.store_transition(observation, actionA, reward, observation_)

			if (step > 200) and (step % 5 == 0):
				playerA.learn()

			# swap observation
			observation = observation_

			# break while loop when end of this episode
			if done:
				break
			step += 1
		reward_a[episode] = total_a
	plt.plot(reward_a)
	plt.ylabel('Cummulative reward')
	plt.xlabel('Episode')
	plt.show()

	# end of game
	print('game over')

def ql_vs_minmax(ql_p, min_p):
	print("ql vs minmax ql")
	vis = Visualiser(env, 80)
	numActions = env.n_actions
	start_time = time.time()
	# no explore

	for episode in range(N_EPISODES):
		# initial observation
		observation = env.reset()
		# print(str(episode))
		if(episode > N_EPISODES - 100):
			vis.update_canvas(env)
		while True:
			# RL choose action based on observation
			actionA = ql_p.choose_action(str(observation))
			actionB = min_p.choose_action(str(observation))

			# RL take action and get next observation and reward
			observation_, reward, done = env.step(actionA, actionB)
			#print(observation_)
			ql_p.learn(str(observation), actionA, reward, str(observation_), done)
			min_p.learn(str(observation), str(observation_), [actionA,actionB], -reward)
			
			observation = observation_
			if(episode > N_EPISODES - 100):
				vis.update_canvas(env)
			if done:
				vis.reset()
				break



if __name__ == '__main__':
	env = Game()
	ql_p = test()
	min_p = test_minmax()
	ql_vs_minmax(ql_p, min_p)
	# test_minmax()
	# test_DQL()
	# testB(cont=True, filenames=("actions", "actionsB"))
	# run_optimal()
	# run_optimalB()