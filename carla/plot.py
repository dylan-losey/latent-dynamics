import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import glob
import numpy as np

# sns.set()

DIR = ''
SUBDIRS = ['oracle.txt', 'sacbaseline.csv', 'ours.csv']
LEGEND_LABELS = ['Ours', 'SAC', 'Oracle']
COLORS = sns.color_palette("husl", n_colors=6)[:3]

ax = plt.gca()

for i, sd in enumerate(SUBDIRS):
	# dir_path = os.path.join(DIR, sd)

	csvs = glob.glob(SUBDIRS[i])
	rews = []
	eps = []
	for c in csvs:
		df = pd.read_csv(c)
		rews.append(df['training/episode-reward-mean'].values)
		eps.append(df['sampler/episodes'].values)

	num_points = 150		# set this
	episode_length = 10		# set this

	rews = [r[:num_points] for r in rews]
	eps = [e[:num_points] for e in eps]

	smoothness = 5
	smoothed = np.mean(np.array(rews).reshape(len(rews), -1, smoothness), axis=2)

	# sns.tsplot(time=eps[0][::smoothness] * episode_length, data=smoothed, color=COLORS[i])

plt.legend(labels=LEGEND_LABELS)
plt.xlabel('Environment Steps')
plt.ylabel('Average Return')
plt.title('Driving')
plt.show()
# plt.savefig('/scr/annie/driving.svg', format='svg')
