import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import numpy as np

def games_won():
	a = [3300, 3197, 2771, 2728, 2631]

	b = [1700, 1803, 2229, 2272, 2369]

	label = [10000, 20000, 40000, 60000, 80000]

	plt.plot(label, a)
	plt.plot(label, b)
	plt.ylabel('Games won')
	plt.xlabel('Episodes')
	plt.show()

def percentage_won():
	N = 5
	a = [34, 37, 45, 46, 48]
	b = [66, 63, 55, 54, 52]
	width = 0.35
	ind = np.arange(N)
	p1 = plt.bar(ind, a, width)
	p2 = plt.bar(ind, b, width,
             bottom=a)
	plt.ylabel('% of games won')
	plt.xlabel('number of episodes trained')
	plt.title('Percentage of games won over after x episodes of training')
	plt.xticks(ind, ('10000', '20000', '40000', '60000', '80000'))
	plt.yticks(np.arange(0, 100, 10))
	plt.legend((p1[0], p2[0]), ('MinMax Q-Learning', 'Q-Learning'))

	plt.show()
percentage_won()