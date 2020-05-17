import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle


def moving_average(a, n=1000) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_visits(data_thetas):
    visits = [0] * 8
    thetas = np.linspace(0, 2*np.pi - np.pi/4, 8)
    for item in np.array(data_thetas):
        dist = -abs(thetas - item)
        idx = np.argmax(dist)
        visits[idx] += 1
    visits = np.array(visits)
    visits = visits * 1.0 / np.sum(visits)
    return visits

def plot_visits(intensities, color, radius, axis):
    thetas = np.linspace(0, 2*np.pi - np.pi/4, 8)
    for count, theta in enumerate(thetas):
        x = (radius*np.cos(theta), radius*np.sin(theta))
        alpha = intensities[count]
        circle = plt.Circle(x, radius=0.1, alpha=alpha, color=color)
        axis.add_artist(circle)

def main():

    naive = pickle.load(open("results/naive.pkl", "rb"))
    dqn_greedy1 = pickle.load(open("results/dqn-greedy1.pkl", "rb"))
    dqn_greedy2 = pickle.load(open("results/dqn-greedy2.pkl", "rb"))
    dqn_greedy3 = pickle.load(open("results/dqn-greedy3.pkl", "rb"))
    dqn_influence1 = pickle.load(open("results/dqn-influence1.pkl", "rb"))
    dqn_influence2 = pickle.load(open("results/dqn-influence2.pkl", "rb"))
    dqn_influence3 = pickle.load(open("results/dqn-influence3.pkl", "rb"))
    ours_greedy1 = pickle.load(open("results/ours-greedy1.pkl", "rb"))
    ours_greedy2 = pickle.load(open("results/ours-greedy2.pkl", "rb"))
    ours_greedy3 = pickle.load(open("results/ours-greedy3.pkl", "rb"))
    ours_influence1 = pickle.load(open("results/ours-influence4.pkl", "rb"))
    ours_influence2 = pickle.load(open("results/ours-influence5.pkl", "rb"))
    ours_influence3 = pickle.load(open("results/ours-influence6.pkl", "rb"))

    naive_score = np.array(naive)
    dqn_greedy_score = (np.array(dqn_greedy1[0]) + np.array(dqn_greedy2[0]) + np.array(dqn_greedy3[0])) / 3.0
    dqn_influence_score = (np.array(dqn_influence1[0]) + np.array(dqn_influence2[0]) + np.array(dqn_influence3[0])) / 3.0
    ours_greedy_score = (np.array(ours_greedy1[0]) + np.array(ours_greedy2[0]) + np.array(ours_greedy3[0])) / 3.0
    ours_influence_score = (np.array(ours_influence1[0]) + np.array(ours_influence2[0]) + np.array(ours_influence3[0])) / 3.0

    plt.plot(moving_average(naive_score))
    plt.plot(moving_average(dqn_greedy_score))
    plt.plot(moving_average(dqn_influence_score))
    plt.plot(moving_average(ours_greedy_score))
    plt.plot(moving_average(ours_influence_score))
    plt.xlabel("Task Number (Episodes)")
    plt.ylabel("Task Reward")
    plt.legend(("dqn-naive","dqn-greedy","dqn-influence","ours-greedy","ours-influence"))
    plt.show()

    dqn_greedy_thetas = (get_visits(dqn_greedy1[1]) + get_visits(dqn_greedy2[1]) + get_visits(dqn_greedy3[1])) / 3.0
    dqn_influence_thetas = (get_visits(dqn_influence1[1]) + get_visits(dqn_influence2[1]) + get_visits(dqn_influence3[1])) / 3.0
    ours_greedy_thetas = (get_visits(ours_greedy1[1]) + get_visits(ours_greedy2[1]) + get_visits(ours_greedy3[1])) / 3.0
    ours_influence_thetas = (get_visits(ours_influence1[1]) + get_visits(ours_influence2[1]) + get_visits(ours_influence3[1])) / 3.0

    # print(ours_influence_thetas)

    fig, ax = plt.subplots()
    plot_visits(dqn_greedy_thetas, color="orange", radius=0.75, axis=ax)
    plot_visits(dqn_influence_thetas, color="red", radius=1.0, axis=ax)
    plot_visits(ours_greedy_thetas, color="green", radius=1.25, axis=ax)
    plot_visits(ours_influence_thetas, color="purple", radius=1.5, axis=ax)
    plt.axis([-2,2,-2,2])
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
