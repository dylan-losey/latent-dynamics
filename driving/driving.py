import numpy as np
import pygame
import torch



class Opponent(pygame.sprite.Sprite):

    def __init__(self, x, y):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,20))
        self.image.fill((135, 206, 235))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = x
        self.y = y
        self.rect.x = (self.x * 150) + 200 - self.rect.size[0] / 2
        self.rect.y = (self.y * 300) + 50 - self.rect.size[1] / 2

    def update(self, x, y):
        self.x = x
        self.y = y
        self.rect = self.image.get_rect(center=self.rect.center)
        self.rect.x = (self.x * 150) + 200 - self.rect.size[0] / 2
        self.rect.y = (self.y * 300) + 50 - self.rect.size[1] / 2


class Player(pygame.sprite.Sprite):

    def __init__(self, x, y):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,20))
        self.image.fill((255, 128, 0))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = x
        self.y = y
        self.rect.x = (self.x * 150) + 200 - self.rect.size[0] / 2
        self.rect.y = (self.y * 300) + 50 - self.rect.size[1] / 2

    def update(self, x, y):
        self.x = x
        self.y = y
        self.rect = self.image.get_rect(center=self.rect.center)
        self.rect.x = (self.x * 150) + 200 - self.rect.size[0] / 2
        self.rect.y = (self.y * 300) + 50 - self.rect.size[1] / 2



# driving environment
class Environment(object):

    def __init__(self):
        self.mode = 0
        self.state = np.array([0.0, 0.0, 0.0, 0.3])
        self.timestep = 0
        pygame.init()
        self.clock = pygame.time.Clock()
        self.world = pygame.display.set_mode([400,400])
        self.agent1 = Player(self.state[0], self.state[1])
        self.agent2 = Opponent(self.state[2], self.state[3])
        self.sprite_list = pygame.sprite.Group()
        self.sprite_list.add(self.agent1)
        self.sprite_list.add(self.agent2)


    # resets agent to start, updates target position
    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.3])
        self.timestep = 0
        return np.array(self.state)

    # returns next state, reward, done, and info
    def step(self, action):
        self.timestep += 1
        # action for ego
        ego_action = [0.0, 0.1]
        if action == 1 and self.state[0] < 0.05:
            ego_action = [0.1, 0.1]
        elif action == 2 and self.state[0] > -0.05:
            ego_action = [-0.1, 0.1]
        # action for other
        other_action = [0.0, 0.05]
        if self.timestep == 5:
            if self.mode == -1:
                other_action = [-0.1, 0.0]
            elif self.mode == 0:
                other_action = [0.0, 0.0]
            elif self.mode == 1:
                other_action = [0.1, 0.0]
        # next state
        deltax = np.array(ego_action + other_action)
        next_state = np.array(self.state + deltax)
        # reward for current state
        reward = 0.0
        if np.linalg.norm(self.state[0:2] - self.state[2:4]) < 0.025:
            reward = -100.0
        # done if trajectory reaches full length
        if self.timestep == 6:
            if self.state[0] < -0.05:
                self.mode = -1
            elif self.state[0] > 0.05:
                self.mode = 1
            else:
                self.mode = 0
        if self.timestep == 10:
            done = True
        else:
            done = False
            self.state += deltax
        # mode if we reset from current state (info)
        info = self.mode
        return next_state, reward, done, [info*1.0]

    def render(self):

        self.agent1.update(self.state[0], self.state[1])
        self.agent2.update(self.state[2], self.state[3])

        # animate
        self.world.fill((255,255,255))
        pygame.draw.rect(self.world, (0, 0, 0), (150, 0, 100, 400), 0)
        self.sprite_list.draw(self.world)
        pygame.display.flip()
        self.clock.tick(5)

    def close(self):
        pygame.quit()
