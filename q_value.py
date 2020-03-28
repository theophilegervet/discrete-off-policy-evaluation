import torch
from abc import ABC, abstractmethod


class QValue(ABC):
    """
    Discrete Q value.
    """
    @abstractmethod
    def Qs(self, state):
        """
        Evaluate Q-values of all actions in a state.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :returns: Q-values torch.Tensor (batch_size, num_actions)
        """
        pass
    
    @abstractmethod
    def Q(self, state, action):
        """
        Evaluate Q-value of one action in a state.
        
        :param state: torch.Tensor (batch_size, state_dim)
        :param action: torch.Tensor (batch_size)
        :returns: Q-value torch.Tensor (batch_size)
        """
        pass