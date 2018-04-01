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
import gym

VIS = False
N_EPISODES = 5000

# test QL vs Random
def test(cont=False, filename=None):

	# reward_a = np.zeros(N_EPISODES)
	# total_a = 0
	numActions = env.n_actions
	playerA = QLearn(actions=list(range(numActions)), reward_decay=0.7)
	playerB = QLearn(actions=list(range(numActions)), reward_decay=0.7)
	# playerB = RandomPlayer(numActions-1)
	if(cont):
		print('loading actions')
		playerA.load_Qtable(filename)
		# playerA.save_Qtable("old_actions")
	# playerB.load_Qtable('qlactions')
	playerA.save_Qtable("saved_players/QR_base")
	playerB.load_Qtable("saved_players/QR_base")
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
			# total_a += reward

			# RL learn from this transition
			playerA.learn(str(observation), actionA, reward, str(observation_), done)
			# swap observation
			observation = observation_

			# break while loop when end of this episode
			if done:
				break
		# reward_a[episode] = total_a
	# plt.plot(reward_a)
	# plt.ylabel('Cummulative reward')
	# plt.xlabel('Episode')
	# plt.show()
	playerA.save_Qtable("saved_players/QR")
	return playerA

# test minmaxQL vs Random
def test_minmax(cont=False, filename=None):

	# reward_a = np.zeros(N_EPISODES)
	# total_a = 0

	numActions = env.n_actions
	drawProbability = 0.1
	decay = 10**(-2. / N_EPISODES * 0.05)
	playerA = MinimaxQPlayer(numActions, numActions, decay=decay, expl=0.01, gamma=1-drawProbability)
	playerB = RandomPlayer(numActions-1)

	if(cont):
		print('loading actions')
		playerA.load_Qtable(filename)
	# playerB = QLearn(actions=list(range(numActions)), reward_decay=0.7)
	# playerA.save_Qtable("saved_players/MR_base")
	# playerB.load_Qtable("saved_players/MR_base")
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
			# total_a += reward

			# RL learn from this transition
			playerA.learn(str(observation), str(observation_), [actionA,actionB], reward)
			# swap observation
			observation = observation_

			# break while loop when end of this episode
			if done:
				break
		# reward_a[episode] = total_a
	# plt.plot(reward_a)
	# plt.ylabel('Cummulative reward')
	# plt.xlabel('Episode')
	# plt.show()
	playerA.save_Qtable("MR")
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
	playerA = QLearn(actions=list(range(numActions)))
	playerA.load_Qtable("saved_players/QR")
	playerB = QLearn(actions=list(range(numActions)))
	playerB.load_Qtable("saved_players/QR_base")
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
	playerA = MinimaxQPlayer(numActions, numActions, decay=decay, expl=0.01, gamma=1-drawProbability)
	playerA.load_Qtable("actions")
	playerB = QLearn(actions=list(range(numActions)))
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
	ql_wins = 0
	minmax_wins = 0
	# no explore
	iterations = 10000
	for episode in range(iterations):
		# initial observation
		observation = env.reset()
		# print(str(episode))
		if(episode % 100 == 0):
			print(str(float(episode) / iterations * 100) + "%")
		# if(episode > iterations - 100):
		# 	vis.update_canvas(env)
		while True:
			# RL choose action based on observation
			actionA = ql_p.choose_action(str(observation))
			actionB = min_p.choose_action(str(observation))

			# RL take action and get next observation and reward
			observation_, reward, done = env.step(actionA, actionB)
			if reward == 1:
				ql_wins += 1
			elif reward == -1:
				minmax_wins += 1

			#print(observation_)
			ql_p.learn(str(observation), actionA, reward, str(observation_), done)
			min_p.learn(str(observation), str(observation_), [actionA,actionB], -reward)
			
			observation = observation_
			# if(episode > iterations - 100):
			# 	vis.update_canvas(env)
			if done:
				vis.reset()
				break
		if(episode == 5000):
			print(ql_wins)
			print(minmax_wins)
	return (ql_wins, minmax_wins)


def test_cartpole():

	bins = np.zeros((4,10))
	bins[0] = np.linspace(-4.8,4.8,10)
	bins[1] = np.linspace(-5,5,10)
	bins[2] = np.linspace(-.418,.418,10)
	bins[3] = np.linspace(-5,5,10)

	iterations = 2000
	env = gym.make('CartPole-v0')
	playerA = QLearn(actions=list(range(env.action_space.n)), reward_decay=0.9)
	for i_episode in range(iterations):
		obs = env.reset()
		observation = assign_bins(obs, bins)

		if(i_episode % 100 == 0):
				print(str(float(i_episode) / iterations * 100) + "%")
		while True:
			if(i_episode > iterations - 100):
				env.render()
			action = playerA.choose_action(str(observation))
			obs_, reward, done, info = env.step(action)
			observation_ = assign_bins(obs_, bins)
			# print(observation_)
			playerA.learn(str(observation), action, reward, str(observation_), done)
			observation = observation_
			if done:
				# print("Episode finished")
				break


def assign_bins(obs, bins):
	state = np.zeros(4)
	for i in range(4):
		state[i] = np.digitize(obs[i], bins[i])
	return state


if __name__ == '__main__':
	env = Game()
	# ql_p = test(True, 'saved_players/QR')
	min_p = test_minmax(True, 'MR')
	# ql_wins, minmax_wins = ql_vs_minmax(ql_p, min_p)
	# print(ql_wins)
	# print(minmax_wins)
	# test_minmax()
	# test_DQL()
	# testB(cont=True, filenames=("actions", "actionsB"))
	# run_optimal()
	# run_optimalB()
	# test_cartpole()