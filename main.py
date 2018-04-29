from env import Game
import time
from visualiser import Visualiser
from randomplayer import *
from ql import QLearn
import matplotlib as mpl
from rps import RockPaperScissors
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from minimax_ql import MinimaxQPlayer
from dqn import DeepQNetwork
import gym

VIS = False
N_EPISODES = 1000

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
		# playerB.load_Qtable("saved_players/QR_base")
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
			# playerB.learn(str(observation), actionB, -reward, str(observation_), done)
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
	playerB.save_Qtable("saved_players/QR_base");
	playerA.save_Qtable("saved_players/QR")
	return playerA

# test minmaxQL vs Random
def test_minmax(cont=False, filename=None):

	# reward_a = np.zeros(N_EPISODES)
	# total_a = 0

	numActions = env.n_actions
	drawProbability = 0.1
	decay = 10**(-2. / N_EPISODES * 0.05)
	playerB = MinimaxQPlayer(numActions, numActions, decay=decay, expl=0.01, gamma=1-drawProbability)
	# playerA = RandomPlayer(numActions-1)
	playerA = QLearn(actions=list(range(numActions)), reward_decay=0.7)

	if(cont):
		print('loading actions')
		playerB.load_Qtable(filename)
	# playerB = QLearn(actions=list(range(numActions)), reward_decay=0.7)
	playerA.load_Qtable("saved_players/MR_base")
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
			playerB.learn(str(observation), str(observation_), [actionB,actionA], -reward)
			# playerA.learn(str(observation), actionA, reward, str(observation_), done)
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
	playerA.save_Qtable("saved_players/MR_base");
	playerB.save_Qtable("MR")
	return playerB


	# end of game
	# print "My program took", time.time() - start_time, "to run"
	# print('game over')
	# playerA.save_Qtable("actions")
	# print(playerA.check_convergence("old_actions"))

def run_optimal():
	vis = Visualiser(env, 80)
	numActions = env.n_actions
	playerA = QLearn(actions=list(range(numActions)))
	playerA.load_Qtable("saved_players/QR")
	playerB = QLearn(actions=list(range(numActions)))
	playerB.load_Qtable("saved_players/QR_base")
	for episode in range(500):
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
	numActions = env.n_actions
	drawProbability = 0.1
	decay = 10**(-2. / N_EPISODES * 0.05)
	vis = Visualiser(env, 80)
	numActions = env.n_actions
	playerB = MinimaxQPlayer(numActions, numActions, decay=decay, expl=0.00, gamma=1-drawProbability)
	playerB.load_Qtable("MR")
	playerA = QLearn(actions=list(range(numActions)))
	playerA.load_Qtable("saved_players/MR_base")
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

def ql_vs_minmax(visualise):
	print("ql vs minmax ql")
	numActions = env.n_actions
	drawProbability = 0.1
	decay = 10**(-2. / N_EPISODES * 0.05)
	if(visualise):
		vis = Visualiser(env, 80)
	numActions = env.n_actions
	start_time = time.time()
	ql_wins = 0
	minmax_wins = 0
	playerA = QLearn(actions=list(range(numActions)), reward_decay=0.7)
	playerB = MinimaxQPlayer(numActions, numActions, decay=decay, expl=0.01, gamma=1-drawProbability)
	playerA.load_Qtable('saved_players/QR')
	playerB.load_Qtable("MR")
	# no explore
	iterations = 5000
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
			actionA = playerA.choose_action(str(observation))
			actionB = playerB.choose_action(str(observation))

			# RL take action and get next observation and reward
			observation_, reward, done = env.step(actionA, actionB)
			if reward == 1:
				ql_wins += 1
			elif reward == -1:
				minmax_wins += 1
			
			observation = observation_
			if(visualise):
				vis.update_canvas(env)
			if done:
				if(visualise):
					vis.reset()
				break
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

def test_rps():
	# P = [[0, -25, 50], [25, 0, -5], [-50, 5, 0]]
	game = RockPaperScissors()
	iterations = 1000
	numActions = 3
	drawProbability = 0.1
	decay = 10**(-2. / iterations * 0.05)
	playerA = MinimaxQPlayer(numActions, numActions, decay=decay, expl=0.01, gamma=1-drawProbability)
	playerB = QLearn(actions=list(range(numActions)))
	

	wins = np.zeros(iterations)

	for i in np.arange(iterations):
		if (i % (iterations / 10) == 0):
			print("%d%%" % (i * 100 / iterations))
		game.restart()
		result = -1
		while result == -1:
			state = boardToState()
			actionA = playerA.choose_action(state)
			actionB = playerB.choose_action(state)
			
			result = game.play(actionA, actionB)
			
			reward = resultToReward(result, actionA, actionB)
			newState = boardToState()

			playerA.learn(state, newState, [actionA, actionB], reward)
			playerB.learn(state, actionB, -reward, newState, True)

		wins[i] = result
	# print(wins)
	plotResult(wins)
	plotPolicy(playerA, game)
	print(playerB.q_table)

def boardToState():
        return 0

def resultToReward(result, actionA=None, actionB=None):
        if result >= 0:
            reward = (result*(-2) + 1)
        else:
            reward = 0
        return reward

def plotResult(wins):
        lenWins = len(wins)
        sumWins = (wins == [[0], [1], [-2]]).sum(1)
        print("Wins A : %d (%0.1f%%)" % (sumWins[0], (100. * sumWins[0] / lenWins)))
        print("Wins B : %d (%0.1f%%)" % (sumWins[1], (100. * sumWins[1] / lenWins)))
        print("Draws  : %d (%0.1f%%)" % (sumWins[2], (100. * sumWins[2] / lenWins)))

        plt.plot((wins == 1).cumsum())
        plt.plot((wins == 0).cumsum())
        plt.ylabel('Games won')
        plt.xlabel('Episodes')
        plt.legend(('Q-Learning', 'MinMax Q-Learning'))
        plt.show()

def assign_bins(obs, bins):
	state = np.zeros(4)
	for i in range(4):
		state[i] = np.digitize(obs[i], bins[i])
	return state

def plotPolicy(player, game):
        for state in player.Q:
            print("\n=================")
            game.draw(game.P)
            # print("State value: %s" % player.V[state])
            player.policyForState(state)


if __name__ == '__main__':
	env = Game()
	# ql_p = test(True, 'saved_players/QR')
	# min_p = test_minmax(True, 'MR')
	# ql_wins, minmax_wins = ql_vs_minmax(False)
	# print(ql_wins)
	# print(minmax_wins)
	# test_rps()
	# test_DQL()
	# testB(cont=True, filenames=("actions", "actionsB"))
	run_optimal()
	# run_optimalB()
	# test_cartpole()