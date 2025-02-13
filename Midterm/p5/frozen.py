import numpy as np
import gym

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", map_name = "8x8", is_slippery=False)  # Use is_slippery=False for a deterministic environment

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
num_episodes = 10

# Initialize the Q-table with zeros
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# Q-learning training
for episode in range(num_episodes):
    state = env.reset()[0]
    done = False

    while not done:
        # Choose an action with epsilon-greedy strategy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit

        # Take the selected action and observe the new state and reward
        new_state, reward, done, _, _ = env.step(action)

        # Update the Q-table using the Q-learning update rule
        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + \
            learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]))

        state = new_state

# Evaluate the trained Q-learning agent
num_test_episodes = 10
total_rewards = 0

for _ in range(num_test_episodes):
    state = env.reset()[0]
    done = False

    while not done:
        action = np.argmax(q_table[state, :])  # Choose the best action
        state, reward, done, _, _ = env.step(action)
        total_rewards += reward

average_reward = total_rewards / num_test_episodes
print(f"Average reward over {num_test_episodes} test episodes: {average_reward}")

# Close the environment
env.close()
