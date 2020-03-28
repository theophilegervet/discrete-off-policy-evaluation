import torch
import numpy as np


class Dataset:
    def __init__(self, episodes):
        self.discount = episodes[0].discount
        self.data = {key: torch.cat([ep.data[key] for ep in episodes])
                     for key in episodes[0].data.keys()}
        self.episode_lengths = [len(ep) for ep in episodes]
        
    def __getattr__(self, key):
        if 'data' not in vars(self):
            raise AttributeError
        try:
            return self.data[key]
        except KeyError:
            raise AttributeError
        
    def __len__(self):
        return self.data["state"].size(0)
    
    def batch(self, batch_size):
        idxs = torch.randint(len(self), size=(batch_size,))
        state = self.data["state"][idxs]
        action = self.data["action"][idxs]
        reward = self.data["reward"][idxs]
        next_state = self.data["next_state"][idxs]
        done = self.data["done"][idxs]
        return state, action, reward, next_state, done
        
    def add(self, episode):
        for key in episode.data.keys():
            self.data[key] = torch.cat([self.data[key], episode.data[key]])
        self.episode_lengths.append(len(episode))