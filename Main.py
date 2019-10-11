import cv2
import time
import pickle
import numpy as np
from PIL import Image
from matplotlib import style
import matplotlib.pyplot as plt
from numpy.random import randint, uniform, random

style.use('ggplot')

SIZE = 10
EPISODES = 10
MOVE_PENALTY = 1
ENEMY_PENALTY = 400
FOOD_REWARD = 100
epsilon = 0.0  # randomness
EPSILON_DECAY = 0.9998
SHOW_EVERY = 1

start_q_table = 'q_table-1570746238.pickle'  # or we can give the file name

LEARNING_RATE = 0.1
DISCOUNT = 0.95

AGENT = 1
FOOD = 2
ENEMY = 3

objects = {
    1: (255, 175, 0),
    2: (0, 255, 0),
    3: (0, 0, 255)
}


class Blob:

    def __init__(self):
        self.x = randint(0, SIZE)
        self.y = randint(0, SIZE)

    def __str__(self):
        return '{}, {}'.format(self.x, self.y)

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        if not x:
            self.x += randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1


if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE + 1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, 'rb') as file:
        q_table = pickle.load(file)

episode_rewards = []
for episode in range(EPISODES):
    agent = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        print('''
        On episode {} epsilon is {}
        {} ep mean {}
        '''.format(episode, epsilon, SHOW_EVERY, np.mean(episode_rewards[-SHOW_EVERY:])))
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        observation = (agent - food, agent - enemy)
        if random() > epsilon:
            action = np.argmax(q_table[observation])
        else:
            action = randint(0, 4)

        agent.action(action)

        enemy.move()

        if agent.x == enemy.x and agent.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif agent.x == food.x and agent.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        new_observation = (agent - food, agent - enemy)
        max_future_q = np.max(q_table[new_observation])
        current_q = q_table[observation][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[observation][action] = new_q

        '''
        now that we know the reward, first we need to observation immediately
        after the move
        '''
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[agent.x][agent.y] = objects[AGENT]
            env[enemy.x][enemy.y] = objects[ENEMY]
            env[food.x][food.y] = objects[FOOD]

            img = Image.fromarray(env, 'RGB')
            img = img.resize((500, 500))
            cv2.imshow('Q Learning First Project', np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPSILON_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel('reward {}ma'.format(SHOW_EVERY))
plt.xlabel('episode number')
plt.show()

with open('q_table-{}.pickle'.format(int(time.time())), 'wb') as file:
    pickle.dump(q_table, file)
