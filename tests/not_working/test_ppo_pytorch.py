import time

import gymnasium
import pygame

import flappy_bird_gymnasium
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]


class ActorCriticPPO(nn.Module):

    def __init__(self, state_size, action_size, hidden_size):
        super(ActorCriticPPO, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, env):
        self.ts_rewards = None
        self.env = env

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.hidden_size = 128

        self.gamma = 0.9  # discount factor
        self.max_steps = 4000  # update policy every n timesteps
        self.epochs = 20  # update policy for epochs
        self.eps_clip = 0.1  # clip parameter for PPO

        self.lr_actor = 3e-4  # learning rate for actor network -- try higher 10^(-2)
        self.lr_critic = 1e-3  # learning rate for critic network
        self.eps = np.finfo(np.float32).eps.item()

        self.buffer = RolloutBuffer()

        self.policy = ActorCriticPPO(self.state_size, self.action_size, self.hidden_size)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])

        self.policy_old = ActorCriticPPO(self.state_size, self.action_size, self.hidden_size)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):

        state = torch.FloatTensor(state)
        action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def compute_returns(self):
        returns = []
        R = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if is_terminal:
                R = 0
                reward = -10
            R = reward + (self.gamma * R)
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        return returns

    def update(self, returns):

        weight = 0.1

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()

        advantages = returns.detach() - old_state_values.detach()

        for _ in range(self.epochs):
            logprobs, state_values, entropy = self.policy.evaluate(old_states,
                                                                   old_actions)  # Evaluating old actions and values
            state_values = torch.squeeze(state_values)  # match state_values tensor dimensions with rewards tensor
            ratios = torch.exp(logprobs - old_logprobs.detach())  # Finding the ratio (pi_theta / pi_theta__old)

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - weight * entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())  # Copy new weights into old policy
        self.buffer.clear()  # clear buffer

    def train(self, model, episodes):
        episode = 0
        self.ts_rewards = []

        state, _ = env.reset(seed=0)

        while episode <= episodes:
            state, _ = env.reset(seed=0)
            state = np.expand_dims(state, axis=0)

            score = 0
            done = False
            time_step = 0
            while not done and time_step <= self.max_steps:
                # select action with policy
                action = self.select_action(state)
                next_state, reward, done,_, info = env.step(action)
                # env.render()
                # time.sleep(0.01)

                model.buffer.rewards.append(reward)
                model.buffer.dones.append(done)

                score += reward

                state = next_state
                state = np.expand_dims(state, axis=0)

                time_step += 1

            returns = self.compute_returns()
            self.update(returns)
            self.ts_rewards.append(score)

            if episode % 100 == 0:
                print("Episode : {} \t\t Timestep : {} \t\t Total Reward : {}".format(episode, time_step, score))
            episode += 1

        env.close()

        self.ts_rewards = savgol_filter(self.ts_rewards, window_length=10, polyorder=3)
        plt.plot(self.ts_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("PPO-Clip rewards over episodes")
        plt.show()

        return self.ts_rewards


if __name__ == "__main__":
    env = flappy_bird_gymnasium.FlappyBirdEnvSimple()

    model = PPO(env)
    model.train(model, 2000)
