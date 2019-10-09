import cv2
import time
import pickle
import numpy as np
from PIL import Image
from matplotlib import style
import matplotlib.pyplot as plt
from numpy.random import randint


style.use('ggplot')

SIZE = 10
EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPSILON_DECAY = 0.9998
SHOW_EVERY = 3000

start_q_table = None  # or we can give the file name

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_NR = 0
FOOD_NR = 1
ENEMY_NR = 2

objects = {
    0: (255, 175, 0),
    1: (0, 255, 0),
    2: (0, 0, 255)
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
        if choice == 2:
            self.move(x=-1, y=1)
        if choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=None, y=None):
        if not x:
            self.x = randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y = randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, 'rb') as file:
        q_table = pickle.load(file)

episode_rewards = []
for episode in range(EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        print('''On episode {} epsilon is {}
        {} ep mean {}
        '''.format(episode, epsilon, SHOW_EVERY, np.mean(episode_rewards[-SHOW_EVERY:])))
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        observation = (player-food, player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[observation])
        else:
            action = randint(0, 4)

        player.action(action)

        # enemy.move()

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
            new_q = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
            new_q = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
            new_q = -MOVE_PENALTY

        new_observation = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_observation])
        current_q = q_table[observation][action]

        q_table[observation][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[player.y][player.x] = objects[PLAYER_NR]
            env[food.y][food.x] = objects[FOOD_NR]
            env[enemy.y][enemy.x] = objects[ENEMY_NR]

            img = Image.fromarray(env, 'RGB')
            img = img.resize((400, 400))
            cv2.imshow('image', np.array(img))
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
