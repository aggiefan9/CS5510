import gym
import torch
import torch.nn as nn
from collections import namedtuple
from itertools import count
import torch.nn.functional as F

# Define the DQN class (you can reuse your existing DQN class definition)
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Define the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)


# Determine the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model and move it to the appropriate device
saved_model_path = 'final_policy_net.pth'  # Change this to the path of your saved model
model = DQN(n_observations, n_actions).to(device)  # Create an instance of your DQN model
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.eval()  # Set the model to evaluation mode

# Define a namedtuple to represent a transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Function to test the model
def test_model(model, max_episodes=10):
    for episode in range(max_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0

        for t in count():
            env.render()  # Display the environment (optional)

            with torch.no_grad():
                action = model(state).max(1)[1].view(1, 1)

            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            total_reward += reward

            if done:
                print(f"Episode {episode + 1} finished after {t + 1} timesteps with a total reward of {total_reward}")
                break

            state = next_state

    env.close()  # Close the environment (optional)

# Run the testing function
test_model(model)
