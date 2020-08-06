import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import csv

def get_modes(filename):
    with open(filename) as f:
        content = []
        for line in f:
            content += [float(line)]
    content = np.asarray(content)
    content = content[:2400]
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



def get_success(filename1, filename2):
    with open(filename1, 'r') as f:
        content = []
        for line in f:
            content += [float(line)]
    content = np.asarray(content)
    with open(filename2, 'r') as f:
        interaction = []
        for line in f:
            interaction += [float(line)]
    interaction = np.asarray(interaction)
    print("I got the success for this many interactions: ", len(interaction))
    return interaction, content

def process_success(content, window):
    # yhat = savgol_filter(content, 1001, 5)
    # return yhat, 0
    filter_mean = []
    filter_std = []
    for idx in range(window, len(content) - window):
        filter_mean += [np.mean(content[idx-window:idx+window])]
        filter_std += [np.std(content[idx-window:idx+window])]
    filter_mean = np.asarray(filter_mean)
    filter_std = np.asarray(filter_std) / np.sqrt(2*window)
    return filter_mean, filter_std



# def test(filename):
#     with open(filename, 'rb') as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         line_count = 0
#         for row in csv_reader:
#             print(row)


file1 = 'oracler.txt'
file2 = 'sacr.txt'
file3 = 'oursr.txt'
file4 = 'slacr.txt'
file5 = 'lilacr.txt'
t1, c1 = get_success('oracler.txt', 'oraclet.txt')
t2, c2 = get_success('sacr.txt', 'sact.txt')
t3, c3 = get_success('slacr.txt', 'slact.txt')
t4, c4 = get_success('lilacr.txt', 'lilact.txt')
t5, c5 = get_success('oursr.txt', 'ourst.txt')
c1, s1 = process_success(c1, 50)
c2, s2 = process_success(c2, 50)
c3, s3 = process_success(c3, 50)
c4, s4 = process_success(c4, 5)
c5, s5 = process_success(c5, 50)
plt.plot(t1[50:-50], c1)
plt.fill_between(t1[50:-50], c1 - s1, c1 + s1, alpha=0.2)
plt.plot(t2[50:-50], c2)
plt.fill_between(t2[50:-50], c2 - s2, c2 + s2, alpha=0.2)
plt.plot(t3[50:-50], c3)
plt.fill_between(t3[50:-50], c3 - s3, c3 + s3, alpha=0.2)
plt.plot(t4[5:-5], c4)
plt.fill_between(t4[5:-5], c4 - s4, c4 + s4, alpha=0.2)
plt.plot(t5[50:-50], c5)
plt.fill_between(t5[50:-50], c5 - s5, c5 + s5, alpha=0.2)
plt.xlim([0, 10500])
plt.show()


plt.plot(m1)
plt.fill_between(range(len(m1)), m1 - s1, m1 + s1, alpha=0.2)
plt.plot(m2)
plt.fill_between(range(len(m2)), m2 - s2, m2 + s2, alpha=0.2)
plt.plot(m3)
plt.fill_between(range(len(m3)), m3 - s3, m3 + s3, alpha=0.2)
plt.plot(m4)
plt.fill_between(range(len(m4)), m4 - s4, m4 + s4, alpha=0.2)
plt.plot(m5)
plt.fill_between(range(len(m5)), m5 - s5, m5 + s5, alpha=0.2)
plt.show()


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
