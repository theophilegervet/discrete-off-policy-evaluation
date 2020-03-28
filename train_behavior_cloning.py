import gym
import torch
import pickle
import argparse
import numpy as np

from fqe import FQE
from dataset import Dataset
from episode import Episode, Transition
from behavior_cloning import BehaviorCloning
from offline_evaluation import evaluate_offline


def collect_episode(bc, env):
    """
    :param bc: BehaviorCloning
    :param env: OpenAI gym environment
    """
    state = env.reset()
    episode = Episode(env.discount)
    done = False

    while not done:
        action, action_prob = bc.single_action(state)
        next_state, reward, done, _ = env.step(action)
        transition = Transition(state, action, action_prob,
                                reward, next_state, done)
        state = next_state
        episode.insert(transition)
        
    return episode
    
    
@torch.no_grad()
def evaluate_online(bc, env, num_episodes=5):
    """
    :param bc: BehaviorCloning
    :param env: OpenAI gym environment
    :param num_episodes: number of evaluation episodes
    """
    episodes = [collect_episode(bc, env) for _ in range(num_episodes)]
    data = Dataset(episodes)
    return {"Score": np.mean([ep.score for ep in episodes])}


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    parser.add_argument('--data_path', type=str, default='data/dqn_CartPole-v0.pkl')
    args = parser.parse_args()
    
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    
    env = gym.make(args.env_name)
    env.discount = data.discount
    
    bc = BehaviorCloning(env.observation_space.shape[0], env.action_space.n)

    # Train
    print(f"Before training: {evaluate_online(bc, env)}")
    bc.train(data, steps=1000)
    print(f"After training: {evaluate_online(bc, env)}")
    
    # Evaluate offline 
    fqe = FQE(bc.state_dim, bc.num_actions)
    fqe.train(data, bc, steps=10000)
    print(f"Offline evaluation: {evaluate_offline(bc, data, fqe)}")