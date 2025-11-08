import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(2, 32, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 32, 64), nn.ReLU()
        )

        self.critic = nn.Sequential(
            nn.Linear(64, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(64, 4), nn.Softmax(dim=-1)
        )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_probs = self.actor(self.network(state))
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        shared = self.network(state)
        action_probs = self.actor(shared)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(shared)
        
        return action_logprobs, state_values, dist_entropy
