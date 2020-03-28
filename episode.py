import torch


class Episode:
    def __init__(self, discount):
        """
        Sequence of Transitions.
        
        :param discount: discount factor to use when calculating total returns
        """
        self.transitions = []
        self.discount = discount
        self.is_complete = False
        
    def __getattr__(self, key):
        if 'data' not in vars(self):
            raise AttributeError
        try:
            return self.data[key]
        except KeyError:
            raise AttributeError
        
    def __len__(self):
        return len(self.transitions)

    def insert(self, transition):
        self.transitions.append(transition)
        if transition.done:
            self.is_complete = True
            self._compute_discounted_returns()
            self._store_tensors()
            self.score = sum([t.reward for t in self.transitions])
            self.discounted_return = self.transitions[0].discounted_return

    def _compute_discounted_returns(self):
        """
        Compute the discounted returns for all the transitions in the episode.
        """
        for t in reversed(range(len(self))):
            self.transitions[t].data["discounted_return"] = self.transitions[t].reward
            if not self.transitions[t].done:
                x = self.transitions[t + 1].discounted_return
                self.transitions[t].data["discounted_return"] += self.discount * x 
                
    def _store_tensors(self):
        self.data = {key: torch.tensor([t.data[key] for t in self.transitions],
                                       dtype=torch.float)
                     for key in self.transitions[0].data.keys()}
        self.data["action"] = self.data["action"].long()


class Transition:
    def __init__(self, state, action, action_prob, reward, next_state, done):
        """
        Episode timestep.
        
        :param state: numpy.ndarray (state_dim)
        :param action: integer in {0, num_actions - 1}
        :param action_prob: probability with which this action was taken 
        :param done: boolean indicating if this is the final transition 
        """
        self.data = {
            "state": state,
            "action": action,
            "behavior_prob": action_prob,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }
    
    def __getattr__(self, key):
        if 'data' not in vars(self):
            raise AttributeError
        try:
            return self.data[key]
        except KeyError:
            raise AttributeError
        
    def __str__(self):
        return f"Transition({self.data})"
    
    def __repr__(self):
        return self.__str__()