import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle


def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():

    naive = pickle.load(open("results/lander_naive.pkl", "rb"))
    dqn = pickle.load(open("results/lander_dqn.pkl", "rb"))
    ours = pickle.load(open("results/lander_ours.pkl", "rb"))

    naive_score = np.array(naive[0])
    dqn_score = np.array(dqn[0])
    ours_score = np.array(ours[0])

    plt.plot(moving_average(naive_score))
    plt.plot(moving_average(dqn_score))
    plt.plot(moving_average(ours_score))
    plt.xlabel("Task Number (Episodes)")
    plt.ylabel("Task Reward")
    plt.legend(("dqn-without-goal","dqn-with-goal","ours"))
    plt.show()

if __name__ == "__main__":
    main()
