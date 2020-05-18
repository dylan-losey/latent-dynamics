import numpy as np
import pygame
import torch



class Opponent(pygame.sprite.Sprite):

    def __init__(self, x, y):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill((128, 128, 128))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = x
        self.y = y
        self.rect.x = (self.x * 150) + 200 - self.rect.size[0] / 2
        self.rect.y = (self.y * 150) + 200 - self.rect.size[1] / 2

    def update(self, x, y):
        self.x = x
        self.y = y
        self.rect = self.image.get_rect(center=self.rect.center)
        self.rect.x = (self.x * 150) + 200 - self.rect.size[0] / 2
        self.rect.y = (self.y * 150) + 200 - self.rect.size[1] / 2


class Player(pygame.sprite.Sprite):

    def __init__(self, x, y):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20))
        self.image.fill((255, 128, 0))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = x
        self.y = y
        self.rect.x = (self.x * 150) + 200 - self.rect.size[0] / 2
        self.rect.y = (self.y * 150) + 200 - self.rect.size[1] / 2

    def update(self, x, y):
        self.x = x
        self.y = y
        self.rect = self.image.get_rect(center=self.rect.center)
        self.rect.x = (self.x * 150) + 200 - self.rect.size[0] / 2
        self.rect.y = (self.y * 150) + 200 - self.rect.size[1] / 2


class Home(pygame.sprite.Sprite):

    def __init__(self, x, y):

        # create sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20))
        self.image.fill((0, 128, 128))
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = x
        self.y = y
        self.rect.x = (self.x * 150) + 200 - self.rect.size[0] / 2
        self.rect.y = (self.y * 150) + 200 - self.rect.size[1] / 2



# inside / outside environment
class Environment(object):

    def __init__(self):
        self.radius = 1.0
        self.theta = 0.0
        self.increment = np.pi/10
        self.goal = self.target(self.theta)
        self.state = np.array([-0.5, 0.5])
        self.timestep = 0
        pygame.init()
        self.world = pygame.display.set_mode([400,400])
        self.home = Home(self.state[0], self.state[1])
        self.agent1 = Player(self.state[0], self.state[1])
        self.agent2 = Opponent(self.goal[0], self.goal[1])
        self.sprite_list = pygame.sprite.Group()
        self.sprite_list.add(self.home)
        self.sprite_list.add(self.agent1)
        self.sprite_list.add(self.agent2)


    # resets agent to start, updates target position
    def reset(self):
        if np.linalg.norm(self.state) > self.radius:
            self.theta += self.increment
        else:
            self.theta -= self.increment
        self.theta = self.theta % (2 * np.pi)
        self.timestep = 0
        self.goal = self.target(self.theta)
        self.state = np.array([-0.5, 0.5])
        return np.array(self.state)

    # target position on circle
    def target(self, theta):
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        return np.array([x, y])

    # returns next state, reward, done, and info
    def step(self, action):
        # next state
        deltax = np.array([0.0, 0.0])
        if action == 1:
            deltax = np.array([0.0, 0.2])
        elif action == 2:
            deltax = np.array([0.0, -0.2])
        elif action == 3:
            deltax = np.array([0.2, 0.0])
        elif action == 4:
            deltax = np.array([-0.2, 0.0])
        next_state = np.array(self.state + deltax)
        # reward for current state
        reward = -np.linalg.norm(self.goal - self.state)**2
        # target if we reset from current state (info)
        theta = self.theta
        if np.linalg.norm(self.state) > self.radius:
            theta += self.increment
        else:
            theta -= self.increment
        info = self.target(theta)
        # done if trajectory reaches full length
        self.timestep += 1
        if self.timestep == 15:
            done = True
        else:
            done = False
            self.state += deltax
        return next_state, reward, done, info

    def render(self):

        self.agent1.update(self.state[0], self.state[1])
        self.agent2.update(self.goal[0], self.goal[1])

        # animate
        self.world.fill((255,255,255))
        pygame.draw.circle(self.world, (0, 0, 0), (200, 200), 150, 1)
        self.sprite_list.draw(self.world)
        pygame.display.flip()

    def close(self):
        pygame.quit()
