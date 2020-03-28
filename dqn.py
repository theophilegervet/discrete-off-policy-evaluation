import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from policy import Policy
from q_value import QValue
from nn_utils import EnsembleLinear


class DQN(Policy, QValue):
    def __init__(self, state_dim, num_actions, hidden_dim=[200], 
                 ensemble_size=1, ensemble_aggregation='mean'):
        """
        Ensemble deep Q network.
        
        :param hidden_dim: list of hidden layer dimensions
        :param ensemble_size: number of ensemble members
        :param ensemble_aggregation: operation on ensemble output
        """
        super(DQN, self).__init__()
        self.net = self._make_network(ensemble_size, state_dim, 
                                      num_actions, hidden_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.ensemble_size = ensemble_size
        assert ensemble_aggregation in ['mean', 'min']
        self.ensemble_aggregation = ensemble_aggregation
        
    def _make_network(self, ensemble_size, input_dim, output_dim, hidden_dim):
        layers = []
        for in_dim, out_dim in zip([input_dim] + hidden_dim, hidden_dim):
            layers.append(EnsembleLinear(ensemble_size, in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(EnsembleLinear(ensemble_size, hidden_dim[-1], output_dim))
        return nn.Sequential(*layers)
        
    def Qs(self, state):
        """
        Evaluate Q-values of all actions in a state.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :returns: Q-values torch.Tensor (batch_size, num_actions)
        """
        ensemble_Qs = self.net(state.repeat(self.ensemble_size, 1, 1))
        if self.ensemble_aggregation == 'min':
            return ensemble_Qs.min(dim=0)[0]
        elif self.ensemble_aggregation == 'mean':
            return ensemble_Qs.mean(dim=0)
    
    def Q(self, state, action):
        """
        Evaluate Q-value of one action in a state.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :param action: torch.Tensor (batch_size)
        :returns: Q-value torch.Tensor (batch_size)
        """
        return self.Qs(state).gather(1, action.view(-1, 1)).view(-1)
    
    @torch.no_grad()
    def max_Q(self, state):
        """
        Evaluate maximum Q-value in a state.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :returns: maximum Q-value torch.Tensor (batch_size)
        """
        return self.Qs(state).max(dim=1)[0]
    
    @torch.no_grad()
    def action(self, state, eps=0):
        """
        Take an action in a state.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :param eps: rate of epsilon greedy exploration
        :returns: action torch.Tensor (batch_size)
        """
        batch_size = state.size(0)
        if random.random() > eps:
            return self.Qs(state).max(dim=1)[1]
        else:
            return torch.randint(high=self.num_actions, size=(batch_size,))
    
    @torch.no_grad()
    def action_probs(self, state, eps=0):
        """
        Evaluate probability of taking each action.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :param eps: rate of epsilon greedy exploration
        :returns: action probability torch.Tensor (batch_size, num_actions)
        """
        batch_size = state.size(0)
        probs = torch.ones(batch_size, self.num_actions) * eps / self.num_actions
        action = self.Qs(state).max(dim=1)[1]
        probs[torch.arange(batch_size), action] += 1 - eps
        return probs
    
    @torch.no_grad()
    def action_prob(self, state, action, eps=0):
        """
        Evaluate probability of taking an action.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :param action: torch.Tensor (batch_size)
        :param eps: rate of epsilon greedy exploration
        :returns: action probability torch.Tensor (batch_size)
        """
        return self.action_probs(state, eps).gather(1, action.view(-1, 1)).view(-1)
    
    def train(self, data, steps=100, batch_size=100):
        """
        :param data: Dataset
        :param steps: number of SGD steps
        :param batch_size: batch size for each SGD step
        """
        assert batch_size % self.ensemble_size == 0
        losses = []

        for _ in range(steps):
            state, action, reward, next_state, done = data.batch(batch_size)

            target_Q = self.max_Q(next_state)
            target_Q = reward + (1 - done) * data.discount * target_Q
            
            # Split batch among ensemble members
            state = state.view(
                self.ensemble_size, batch_size // self.ensemble_size, -1)
            Qs = self.net(state).view(batch_size, -1)
            current_Q = Qs.gather(1, action.view(-1, 1)).view(-1)

            loss = F.mse_loss(current_Q, target_Q)
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return losses