import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from flappy_bird_gymnasium.envs.flappy_bird_env_simple import FlappyBirdEnvSimple


# Define the PPO agent network
class PPO(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class PPOAgent:
    def __init__(self):
        self.env = FlappyBirdEnvSimple()
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.hidden_size = 128

        self.ppo = PPO(self.state_size, self.action_size, self.hidden_size)
        self.optimizer = optim.Adam(self.ppo.parameters(), lr=0.001)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.ppo(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()  # , dist.log_prob(action)

    def compute_returns(self, memory, rewards, gamma):
        R = 0
        returns = []
        for reward, is_terminal in zip(reversed(rewards), reversed(memory)):
            if is_terminal:
                R = 0
                reward = -10
            R = reward + gamma * R
            returns.insert(0, R)
        return returns

    def update(self, memory, returns, clip_value):
        states = torch.from_numpy(np.vstack([m[0] for m in memory])).float()
        actions = torch.LongTensor([m[1] for m in memory])
        old_logprobs = torch.FloatTensor([m[2] for m in memory])
        returns = torch.FloatTensor(returns)
        values = torch.FloatTensor([m[4] for m in memory])

        advantage = returns - values

        for _ in range(30):
            new_logprobs = self.ppo(states)
            new_dist = Categorical(new_logprobs)
            new_logprobs = new_dist.log_prob(actions)

            ratio = (new_logprobs - old_logprobs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_value, 1.0 + clip_value) * advantage

            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(returns, values)

            loss = 0.5 * critic_loss + actor_loss - 0.1 * new_dist.entropy().mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.ppo.load_state_dict(self.ppo.state_dict())

        # states = torch.FloatTensor([m[0] for m in memory])
        # actions = torch.LongTensor([m[1] for m in memory])
        # returns = torch.FloatTensor(returns)
        # advantages = returns - returns.mean()
        #
        # for _ in range(batch_size):
        #     old_probs = agent(states)
        #     old_dist = Categorical(old_probs)
        #     old_log_probs = old_dist.log_prob(actions)
        #
        #     new_probs = agent(states)
        #     new_dist = Categorical(new_probs)
        #     new_log_probs = new_dist.log_prob(actions)
        #
        #     ratio = (new_log_probs - old_log_probs).exp()
        #     surrogate1 = ratio * advantages
        #     surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        #     surrogate_loss = -torch.min(surrogate1, surrogate2).mean()
        #
        #     value = agent(states).gather(1, actions.unsqueeze(1)).squeeze()
        #     value_loss = F.mse_loss(value, returns)
        #
        #     entropy_loss = -(new_probs * new_probs.log()).sum(dim=1).mean()
        #
        #     loss = surrogate_loss + value_coeff * value_loss - entropy_coeff * entropy_loss
        #
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #
        # print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    # Define the PPO-Clip algorithm
    def train(self, epochs, gamma, clip_value):
        for epoch in range(epochs):
            memory = []
            total_reward = 0
            step = 0
            rewards = []
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done = False
            while not done:  # and step<4000:
                action = self.select_action(state)
                next_state, reward, done, _, info = self.env.step(action)

                memory.append((state, action, reward, next_state, done))
                rewards.append(reward)
                total_reward += reward
                state = next_state
                step += 1

            returns = self.compute_returns(memory, rewards, gamma)
            self.update(memory, returns, clip_value)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Total Reward: {total_reward}")


# Create the Flappy Bird environment
env = FlappyBirdEnvSimple()

# Create the PPO agent
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
agent = PPOAgent()
agent.train(epochs=10000, gamma=0.9, clip_value=0.1)
