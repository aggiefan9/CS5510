import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env_id = 'FrozenLake-v1'
env = gym.make(env_id, map_name="8x8")
## The training map is the default 8x8 map, while the eval is a randomly generated one
eval_env = gym.make(env_id, desc=generate_random_map(size=8), render_mode="human")
n_timesteps = 85000
model = PPO("MlpPolicy", env, verbose=0)
model.learn(n_timesteps)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")