import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from torch.utils.data import Dataset, DataLoader
from flappy_bird_gymnasium.envs import FlappyBirdEnvSimple

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# Define the DQN agent
class DQNAgent:
    def __init__(self, env, replay_buffer_capacity=10000, batch_size=64, gamma=0.95, epsilon=0.5,
                 epsilon_decay=0.55, epsilon_min=0.01, target_update_freq=100):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.hidden_size = 64

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq

        # Create the main DQN and target DQN
        self.dqn = DQN(self.state_size, self.action_size, self.hidden_size).to(device)
        self.target_dqn = DQN(self.state_size, self.action_size, self.hidden_size).to(device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        # if random.random() < self.epsilon:
        #     action = random.randint(0, self.action_size - 1)
        # else:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.dqn(state_tensor)
            action = q_values.argmax().item()
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = np.array(states)
        states_tensor = torch.tensor(states_tensor, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states_tensor = np.array(next_states)
        next_states_tensor = torch.tensor(next_states_tensor, dtype=torch.float32).to(device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        current_q_values = self.dqn(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states_tensor).max(1)[0].unsqueeze(1)
            target_q_values = rewards_tensor + self.gamma * (1 - dones_tensor) * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.target_update_freq > 0 and len(self.replay_buffer) % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

    def train(self, num_episodes): #, max_steps):
        self.ts_rewards = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            step = 0
            while not done: # or step<max_steps:
                action = agent.select_action(state)
                next_state, reward, done, _, info = env.step(action)
                agent.replay_buffer.push(state, action, reward, next_state, done)
                agent.update()
                state = next_state
                total_reward += reward
                step += 1

            self.ts_rewards.append(total_reward)
            if episode % 100 == 0:
                print(f"Episode: {episode}, Total Reward: {total_reward}")

        self.ts_rewards = savgol_filter(self.ts_rewards, window_length=50, polyorder=3)
        plt.plot(self.ts_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DQN rewards over episodes")
        plt.show()


if __name__ == '__main__':
    env = FlappyBirdEnvSimple()
    num_episodes = 2000
    # max_steps = 4000
    agent = DQNAgent(env)
    agent.train(num_episodes) #, max_steps)

    # # Test the trained agent
    # state = env.reset()
    # done = False
    # total_reward = 0
    #
    # while not done:
    #     action = agent.select_action(state)
    #     state, reward, done, _ = env.step(action)
    #     total_reward += reward
    #
    # print(f"Total Reward: {total_reward}")
    #
    # # Close the environment
    # env.close()
