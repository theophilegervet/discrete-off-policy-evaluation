import torch
import torch.nn as nn
import torch.nn.functional as F

from policy import Policy


class BehaviorCloning(Policy):
    def __init__(self, state_dim, num_actions, hidden_dim=[200]):
        """
        Discrete behavior cloning policy.
        
        :param hidden_dim: list of hidden layer dimensions
        """
        super(BehaviorCloning, self).__init__()
        self.net = self._make_network(state_dim, num_actions, hidden_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.state_dim = state_dim
        self.num_actions = num_actions
        
    def _make_network(self, input_dim, output_dim, hidden_dim):
        layers = []
        for in_dim, out_dim in zip([input_dim] + hidden_dim, hidden_dim):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        return nn.Sequential(*layers)
    
    @torch.no_grad()
    def action(self, state):
        """
        Take an action in a state.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :returns: action torch.Tensor (batch_size)
        """
        _, action = self.net(state).max(dim=1)
        return action
    
    @torch.no_grad()
    def action_probs(self, state):
        """
        Evaluate probability of taking each action.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :returns: action probability torch.Tensor (batch_size, num_actions)
        """
        return F.softmax(self.net(state), dim=1)
    
    @torch.no_grad()
    def action_prob(self, state, action):
        """
        Evaluate probability of taking an action.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :param action: torch.Tensor (batch_size)
        :returns: action probability torch.Tensor (batch_size)
        """
        return self.action_probs(state).gather(1, action.view(-1, 1)).view(-1)
    
    def train(self, data, steps=100, batch_size=100):
        """
        :param data: Dataset
        :param steps: number of SGD steps
        :param batch_size: batch size for each SGD step
        """
        losses = []
        criterion = nn.CrossEntropyLoss()

        for _ in range(steps):
            state, action, _, _, _ = data.batch(batch_size)

            loss = criterion(self.net(state), action)
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return losses