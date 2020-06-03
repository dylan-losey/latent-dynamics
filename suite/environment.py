import numpy as np
import torch
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.wrappers.ik_wrapper import IKWrapper
import time
import sys


class Franka(object):

    def __init__(self):

        # create environment instance
        self.env = suite.make(
            "PandaILIAD",
            has_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            gripper_visualization=True,
            control_freq=100
        )
        # reset the environment
        obs = self.env.reset()
        self.x = obs['eef_pos']
        # self.env.viewer.set_camera(camera_id=0)
        # enable controlling the end effector directly instead of using joint velocities
        self.env = IKWrapper(self.env)

        self.timestep = 0
        self.theta = np.pi/2
        self.target = np.array([0.4, 0.4*np.sin(self.theta), 0.5])
        self.stepsize = 0.05

    def render(self):
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        self.x = obs['eef_pos']
        # self.env.viewer.set_camera(camera_id=0)
        self.timestep = 0
        self.prev_shaping = None
        return np.array(self.x)

    def step(self, action):
        # record current context
        self.timestep += 1
        state = np.array(self.x)
        dist2target = np.linalg.norm(self.target - state)
        # compute reward and done
        if dist2target < 0.1:
            done = True
            reward = +1.0
            print("we made it!")
        elif self.timestep == 500:
            done = True
            reward = -1.0 * dist2target
        else:
            done = False
            reward = 0.0
            shaping = -1.0 * dist2target
            if self.prev_shaping is not None:
                reward = shaping - self.prev_shaping
            self.prev_shaping = shaping
        # update target
        if done:
            self.theta += np.pi
            self.target = np.array([0.4, 0.4*np.sin(self.theta), 0.5])
        # take action
        dpos = np.array([0.0, 0.0, 0.0])
        dquat = np.array([0.0, 0.0, 0.0, 1.0])
        dgrasp = [0.0]
        if action == 1:
            dpos[0] = self.stepsize
        elif action == 2:
            dpos[0] = -self.stepsize
        elif action == 3:
            dpos[1] = self.stepsize
        elif action == 4:
            dpos[1] = -self.stepsize
        elif action == 5:
            dpos[2] = self.stepsize
        elif action == 6:
            dpos[2] = -self.stepsize
        action = np.concatenate([dpos, dquat, dgrasp])
        obs, _, _, _ = self.env.step(action)
        self.x = obs['eef_pos']
        next_state = np.array(self.x, dtype=np.float32)
        info = np.array(self.target, dtype=np.float32)
        return next_state, reward, done, info



# env = Franka()
# state = env.reset()
# for idx in range(1000):
#     action = np.random.choice(np.arange(7))
#     next_state, reward, done, info = env.step(action)
#     env.render()
