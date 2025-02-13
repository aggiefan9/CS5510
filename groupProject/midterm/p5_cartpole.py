import numpy as np
from cartpole import *
from stable_baselines3 import PPO

env_id = "CartPole-v1"
env = CartPoleEnv()
n_timesteps = 15000
model = PPO("MlpPolicy", env_id, verbose=0)
model.learn(n_timesteps)

def evaluate_and_render(model, env, num_eval_episodes=10):
    all_rewards = []
    for _ in range(num_eval_episodes):
        obs = env.reset()
        done = False
        episode_rewards = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_rewards += reward
            env.render()
        all_rewards.append(episode_rewards)
    env.close()
    
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    
    return mean_reward, std_reward

mean_reward, std_reward = evaluate_and_render(model, env, num_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")