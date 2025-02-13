from cartpole import *

def simple_control_policy(observation):
    # Unpack the observation
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
    if abs(pole_angle) < 0:
        action = 1 if pole_angle < 0 else 0
    else:
        action = 1 if pole_angle > 0 else 0
    return action

# Create the CartPole environment
env = CartPoleEnv()
observation = env.reset()
episode_reward = 0

while True:
    env.render()
    ## Use the control policy to determine the action
    action = simple_control_policy(observation)
    observation, reward, done, _ = env.step(action)
    episode_reward += reward

    if done:
        print(f"Episode Reward: {episode_reward}")
        break


env.close()
