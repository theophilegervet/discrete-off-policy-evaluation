import torch
from abc import ABC, abstractmethod


class Policy(ABC):
    """
    Discrete policy.
    """
    @abstractmethod
    def action(self, state):
        """
        Take an action in a state.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :returns: action torch.Tensor (batch_size)
        """
        pass 
    
    @abstractmethod
    def action_probs(self, state):
        """
        Evaluate probability of taking each action.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :returns: action probability torch.Tensor (batch_size, num_actions)
        """
        pass
    
    @abstractmethod
    def action_prob(self, state, action):
        """
        Evaluate probability of taking an action.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :param action: torch.Tensor (batch_size)
        :returns: action probability torch.Tensor (batch_size)
        """
        pass
    
    def single_action(self, state, *args):
        """
        Take an action when interacting with a gym environment.
        
        :param state: numpy.ndarray (state_dim)
        :returns: action integer
        :returns: action probability float
        """
        tensor_state = torch.tensor(state, dtype=torch.float).view(1, -1)
        tensor_action = self.action(tensor_state, *args)
        tensor_prob = self.action_prob(tensor_state, tensor_action, *args)
        return tensor_action.item(), tensor_prob.item()