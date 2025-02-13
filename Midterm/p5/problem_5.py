import gym
import numpy as np
import tensorflow as tf
from cartpole import *

# Create the CartPole environment
# env = gym.make("CartPole-v1")
env = CartPoleEnv()

# Define your neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, input_shape=(env.observation_space.shape[0],), activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(env.action_space.n, activation='linear')
])
model.compile(optimizer="adam", loss="MeanSquaredError")

# Define Q-learning parameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 0.001

# Train the model using Q-learning
for episode in range(1000):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Select an action using epsilon-greedy strategy
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.predict(state.reshape(1, -1), verbose=0)
            action = np.argmax(q_values)

        next_state, reward, done, _ = env.step(action)

        # Update the Q-value
        target = reward + gamma * np.max(model.predict(next_state.reshape(1, -1), verbose=0))
        q_values = model.predict(state.reshape(1, -1), verbose=0)
        q_values[0][action] = target
        model.fit(state.reshape(1, -1), q_values, verbose=0)

        state = next_state
        total_reward += reward

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode: {episode}, Total Reward: {total_reward}")

# Save the trained model
model.save("cartpole_model.h5")
