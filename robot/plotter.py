import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os


def get_modes(filename):
    with open(filename) as f:
        content = []
        for line in f:
            content += [float(line)]
    content = np.asarray(content)
    content = content[2150:2350]
    print("I got the modes for this many interactions: ", len(content))
    return content

def process_modes(content, low_idx, high_idx):
    bins = [0]*3
    counter = 0
    for idx in range(low_idx, high_idx):
        counter += 1
        if content[idx] == 0:
            bins[1] += 1
        elif content[idx] < 0:
            bins[2] += 1
        else:
            bins[0] += 1
    bins = np.asarray(bins)
    bins = bins / sum(bins)
    print("I looked at this many interactions: ", counter)
    print("Here are the ratios: [left, middle, right]: ", bins)
    return bins

def plot_modes(bins):
    x = [0, 1, 2]
    plt.bar(x, bins, width=0.8, align='center')
    plt.ylim([0, 0.5])
    plt.show()



def get_success(filename):
    with open(filename) as f:
        content = []
        for line in f:
            content += [float(line)]
    content = np.asarray(content)
    content = content[2150:2350]
    print("I got the success for this many interactions: ", len(content))
    return content

def process_success(content):
    # yhat = savgol_filter(content, 1001, 5)
    # return yhat, 0
    window = 100
    filter_mean = []
    filter_std = []
    for idx in range(window, len(content) - window):
        filter_mean += [np.mean(content[idx-window:idx+window])]
        filter_std += [np.std(content[idx-window:idx+window])]
    filter_mean = np.asarray(filter_mean)
    filter_std = np.asarray(filter_std) / np.sqrt(2*window)
    return filter_mean, filter_std


file1 = 'modes_no_influence.csv'
file2 = 'modes_influence.csv'
file3 = 'success_no_influence.csv'
file4 = 'success_influence.csv'
mode1 = get_modes(file1)
mode2 = get_modes(file2)
succ1 = get_success(file3)
succ2 = get_success(file4)

for idx, item in enumerate(mode1):
    if item > 0:
        mode1[idx] = 2.0
    else:
        mode1[idx] = 1.0
for idx, item in enumerate(mode2):
    if item > 0:
        mode2[idx] = 2.0
    else:
        mode2[idx] = 1.0

reward1, reward2 = [], []
for idx in range(len(mode1)):
    reward1 += [mode1[idx] * succ1[idx]]
    reward2 += [mode2[idx] * succ2[idx]]
print(reward1, reward2)
print(np.mean(reward1), np.std(reward1)/np.sqrt(200))
print(np.mean(reward2), np.std(reward2)/np.sqrt(200))
print(np.mean(succ1), np.mean(succ2))

# file1 = 'success_no_influence.csv'
# file2 = 'success_influence.csv'
# file3 = 'success_sac.csv'
# content1 = get_modes(file1)
# content2 = get_modes(file2)
# content3 = get_modes(file3)
# m1, s1 = process_success(content1)
# m2, s2 = process_success(content2)
# m3, s3 = process_success(content3)
# plt.plot(m1)
# plt.fill_between(range(len(m1)), m1 - s1, m1 + s1, alpha=0.2)
# plt.plot(m2)
# plt.fill_between(range(len(m2)), m2 - s2, m2 + s2, alpha=0.2)
# plt.plot(m3)
# plt.fill_between(range(len(m3)), m3 - s3, m3 + s3, alpha=0.2)
# plt.plot([0, len(m1)], [0.39, 0.39])
# plt.show()


# content1 = get_modes(file1)
# content2 = get_modes(file2)
#
# low_idx = 2150
# high_idx = 2350
#
# b1 = process_modes(content1, low_idx, high_idx)
# b2 = process_modes(content2, low_idx, high_idx)
# plot_modes(b1)
# plot_modes(b2)
#
#
