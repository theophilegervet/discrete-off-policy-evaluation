import torch
from torch.nn.utils.rnn import pad_sequence


def evaluate_offline(policy, data, qvalue=None):
    """
    Evaluate a policy offline with importance sampling.
    
    :param policy: Policy
    :param data: Dataset
    :param qvalue: QValue
    """
    
    def dataset2episodes(X, pad):
        """
        :param X: torch.Tensor (len(data), ...)
        :param pad: padding value
        :returns: torch.Tensor (num_episodes, max_episode_length, ...)
        """
        X = torch.split(X, data.episode_lengths)
        X = pad_sequence(X, padding_value=pad)
        return X
    
    estimators = {}
    
    evaluation_prob = policy.action_prob(data.state, data.action)
    ratio = evaluation_prob / data.behavior_prob
    
    reward = dataset2episodes(data.reward, pad=0)
    discount = torch.tensor([data.discount ** t 
                             for t in range(reward.size(0))]).view(-1, 1)
    discounted_reward = reward * discount
    
    ratio_IS = dataset2episodes(ratio, pad=1)
    ratio_IS = torch.prod(ratio_IS, dim=0) + 1e-45
    ep_IS = ratio_IS * torch.sum(discounted_reward, dim=0)
    IS = ep_IS.mean()
    WIS = ep_IS.sum() / ratio_IS.sum()
  
    ratio_PDIS = dataset2episodes(ratio, pad=0)
    ratio_PDIS = torch.cumprod(ratio_PDIS, dim=0) + 1e-45
    ep_PDIS = (ratio_PDIS * discounted_reward).sum(dim=0)
    PDIS = ep_PDIS.mean()
    weighted_ratio_PDIS = ratio_PDIS / ratio_PDIS.sum(dim=-1, keepdim=True)
    WPDIS = (weighted_ratio_PDIS * discounted_reward).sum()
    
    estimators = {"IS": IS.item(),
                  "WIS": WIS.item(),
                  "PDIS": PDIS.item(),
                  "WPDIS": WPDIS.item()}
    
    if qvalue is not None:
        Qs = qvalue.Qs(data.state)
        Q = Qs.gather(1, data.action.view(-1, 1)).view(-1)
        Qs = dataset2episodes(Qs, pad=0)
        Q = dataset2episodes(Q, pad=0)
        
        probs = policy.action_probs(data.state)
        probs = dataset2episodes(probs, pad=0)
        
        ep_direct = (Qs[0] * probs[0]).sum(dim=-1)
        direct = ep_direct.mean()
        
        next_Qs = qvalue.Qs(data.next_state)
        next_Qs = dataset2episodes(next_Qs, pad=0)
        
        next_probs = policy.action_probs(data.next_state)
        next_probs = dataset2episodes(next_probs, pad=0)
        
        next_V = (next_Qs * next_probs).sum(dim=-1)
        
        correction = reward + data.discount * next_V - Q
        discounted_correction = correction * discount
        ep_DR = ep_direct + (ratio_PDIS * discounted_correction).sum(dim=0)
        DR = ep_DR.mean()
        WDR = (weighted_ratio_PDIS * discounted_correction).sum()
        
        estimators.update({"direct": direct.item(),
                           "DR": DR.item(),
                           "WDR": WDR.item()})
    
    return estimators
