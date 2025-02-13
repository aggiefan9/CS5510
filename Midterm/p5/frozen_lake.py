import gym
import numpy as np
import time, pickle, os

env = gym.make('FrozenLake-v1',map_name='8x8',render_mode='human')

epsilon = 0.9
total_episodes = 100
max_steps = 100
decay = 0.95
min_epsilon = 0.05

lr_rate = 0.81
gamma = 0.96

Q = np.zeros((env.observation_space.n, env.action_space.n))
    
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

# Start
for episode in range(total_episodes):
    state = env.reset()[0]
    t = 0
    
    while t < max_steps:
        env.render()

        action = choose_action(state)  

        state2, reward, done, _, info = env.step(action)  
        learn(state, state2, reward, action)

        state = state2

        t += 1
       
        if done:
            break

    epsilon = max(epsilon * decay, min_epsilon)

print(Q)