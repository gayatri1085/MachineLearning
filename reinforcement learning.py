import numpy as np
import random


states = [0, 1, 2]
actions = [0, 1]  # 0: left, 1: right
Q = np.zeros((len(states), len(actions)))

alpha = 0.1 
gamma = 0.9
episodes = 1000

for episode in range(episodes):
    state = random.choice(states)
    done = False

    while not done:
        # Choose action
        if random.uniform(0, 1) < 0.1:  # Exploration
            action = random.choice(actions)
        else:  # Exploitation
            action = np.argmax(Q[state])

        next_state = state + (1 if action == 1 else -1)
        reward = 1 if next_state == 2 else 0
        done = next_state == 2


        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state

print("Trained Q-Table:")
print(Q)
