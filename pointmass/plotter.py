import numpy as np
import matplotlib.pyplot as plt
import os


def get_traj(folder, num):
    lst = os.listdir(folder)
    lst.sort()
    lst = lst[-num*2:]
    traj = []
    for name in lst:
        if name[-1] == 't':
            filename = folder + '/' + name
            with open(filename) as f:
                content = []
                for line in f:
                    currentline = line.split(" ")
                    content += [float(x) for x in currentline]
            content = 10*np.asarray(content)
            content = np.reshape(content,(51,2))
            content[:,1] += 0.5
            curr_traj = content[[1,4,-1],:]
            curr_traj[0,:] = [0,0.5]
            traj.append(curr_traj)
    print("I got the traj for this many epsiodes: ", len(traj))
    return traj

def plot_traj(folder, n_traj, color):
    xi = get_traj(folder, n_traj)
    alpha = np.linspace(0.1, 1, n_traj)
    for idx, item in enumerate(xi):
        plt.plot(item[:,0], item[:,1],'o-',color=[color[0], color[1], color[2], alpha[idx]],MarkerFaceColor=[1,1,1],linewidth=5)


def get_target(folder, num):
    lst = os.listdir(folder)
    lst.sort()
    lst = lst[-num*2:]
    target = []
    for name in lst:
        if name[-1] == 't':
            filename = folder + '/' + name
            with open(filename) as f:
                content = []
                for line in f:
                    currentline = line.split(" ")
                    content += [float(x) for x in currentline]
            content = np.asarray(content)
            content = np.reshape(content,(51,2))
            curr_target = content[0,:].tolist()
            target.append(curr_target)
    target = np.asarray(target)
    print("I got the target for this many epsiodes: ", len(target))
    return target


def get_density(target, n):
    theta = np.linspace(0, 2*np.pi, n+1)[0:n]
    center = np.zeros((n, 2))
    for idx in range(n):
        center[idx,:] = [np.cos(theta[idx]), np.sin(theta[idx])]
    count = [0] * n
    for idx in range(len(target)):
        curr_target = target[idx,:]
        min_dist = np.Inf
        min_idx = None
        for jdx in range(n):
            curr_center = np.asarray([0.1*np.cos(theta[jdx]), 0.1*np.sin(theta[jdx])-0.05])
            curr_dist = np.linalg.norm(curr_target - curr_center)
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_idx = jdx
        count[min_idx] += 1
    count = np.asarray(count)
    count = count / sum(count)
    return count, center

def plot_density(d, c, r, color):
    for idx in range(len(d)):
        alpha = min([d[idx], 1.0])
        circle = plt.Circle((r * c[idx,0], r * c[idx,1]), 0.1, color=[color[0],color[1],color[2],alpha])
        plt.gcf().gca().add_artist(circle)


folder1 = "lilac"
folder2 = "ours-without-influence"
folder3 = "ours-with-influence"
n_steps = 20
n_traj = 4
n_target = 100
theta = np.linspace(0, 2*np.pi, 100)


t1 = get_target(folder1, n_target)
t2 = get_target(folder2, n_target)
t3 = get_target(folder3, n_target)
(d1,c1) = get_density(t1, n_steps)
(d2,c2) = get_density(t2, n_steps)
(d3,c3) = get_density(t3, n_steps)


plt.plot(1.2*np.cos(theta), 1.2*np.sin(theta), 'w-')
plot_density(d1, c1, 1.0, [124./255, 111./255, 145./255])
plot_traj(folder1, n_traj, [124./255, 111./255, 145./255])
plt.axis('equal')
plt.show()


plt.plot(1.2*np.cos(theta), 1.2*np.sin(theta), 'w-')
plot_density(d2, c2, 1.0, [1.0, 160./255, 0])
plot_traj(folder2, n_traj, [1.0, 160./255, 0])
plt.axis('equal')
plt.show()


plt.plot(1.2*np.cos(theta), 1.2*np.sin(theta), 'w-')
plot_density(d3, c3, 1.0, [170./255, 0, 212./255])
plot_traj(folder3, n_traj, [170./255, 0, 212./255])
plt.axis('equal')
plt.show()


# plt.plot(0,0,'kx')
# plt.plot(t1[:,0], t1[:,1],color=[0.5,0.5,0.5,1],linestyle='None',marker='o',markersize=3)
# plt.plot(t2[:,0], t2[:,1],color=[0,0,0.8,1],linestyle='None',marker='o',markersize=5)
# plt.plot(t3[:,0], t3[:,1],color=[1,0.5,0,1],linestyle='None',marker='o',markersize=5)
# plt.axis([-0.2,0.2,-0.2,0.2])
# plt.show()
