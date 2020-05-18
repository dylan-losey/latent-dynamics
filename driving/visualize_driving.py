import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle


def moving_average(a, n=1000) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_changes(data_modes):
    change = [0] * 2
    data_modes = np.array(data_modes)
    for idx in range(len(data_modes) - 5000, len(data_modes)-1):
        curr = data_modes[idx]
        next = data_modes[idx+1]
        if curr == -1 and next == -1:
            change[0] += 1
        elif curr == 0 and next == 0:
            change[0] += 1
        elif curr == 1 and next == 1:
            change[0] += 1
        elif (curr == -1 and next == 0) or (curr == 0 and next == -1):
            change[1] += 1
        elif (curr == 0 and next == 1) or (curr == 1 and next == 0):
            change[1] += 1
        elif (curr == -1 and next == 1) or (curr == 1 and next == -1):
            change[1] += 1
    change = np.array(change)
    change = change * 1.0 / np.sum(change)
    return change


def main():

    naive1 = pickle.load(open("results/driving-naive0.pkl", "rb"))
    naive2 = pickle.load(open("results/driving-naive1.pkl", "rb"))
    naive3 = pickle.load(open("results/driving-naive2.pkl", "rb"))
    dqn1 = pickle.load(open("results/driving-dqn0.pkl", "rb"))
    dqn2 = pickle.load(open("results/driving-dqn1.pkl", "rb"))
    dqn3 = pickle.load(open("results/driving-dqn2.pkl", "rb"))
    ours1 = pickle.load(open("results/driving-ours0.pkl", "rb"))
    ours2 = pickle.load(open("results/driving-ours1.pkl", "rb"))
    ours3 = pickle.load(open("results/driving-ours2.pkl", "rb"))

    naive_score = (np.array(naive1[0]) + np.array(naive2[0]) + np.array(naive3[0])) / 3.0
    dqn_score = (np.array(dqn1[0]) + np.array(dqn2[0]) + np.array(dqn3[0])) / 3.0
    ours_score = (np.array(ours1[0]) + np.array(ours2[0]) + np.array(ours3[0])) / 3.0

    plt.plot(moving_average(naive_score))
    plt.plot(moving_average(dqn_score))
    plt.plot(moving_average(ours_score))
    plt.xlabel("Task Number (Episodes)")
    plt.ylabel("Task Reward")
    plt.legend(("naive","dqn","ours"))
    plt.show()

    naive_mode = (get_changes(naive1[1]) + get_changes(naive2[1]) + get_changes(naive3[1])) / 3.0
    dqn_mode = (get_changes(dqn1[1]) + get_changes(dqn2[1]) + get_changes(dqn3[1])) / 3.0
    ours_mode = (get_changes(ours1[1]) + get_changes(ours2[1]) + get_changes(ours3[1])) / 3.0

    plt.bar([-0.2, 0.8], naive_mode, width=0.2)
    plt.bar([0, 1], dqn_mode, width=0.2)
    plt.bar([0.2, 1.2], ours_mode, width=0.2)
    plt.show()


if __name__ == "__main__":
    main()
