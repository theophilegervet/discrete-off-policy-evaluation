import gym
import torch
import pickle
import argparse
import numpy as np

from dqn import DQN
from dataset import Dataset
from episode import Episode, Transition
from offline_evaluation import evaluate_offline


def collect_episode(dqn, env, eps):
    """
    :param dqn: DQN
    :param env: OpenAI gym environment
    :param eps: rate of epsilon greedy exploration
    """
    state = env.reset()
    episode = Episode(env.discount)
    done = False

    while not done:
        action, action_prob = dqn.single_action(state, eps)
        next_state, reward, done, _ = env.step(action)
        transition = Transition(state, action, action_prob,
                                reward, next_state, done)
        state = next_state
        episode.insert(transition)
        
    return episode
    
    
@torch.no_grad()
def evaluate_online(dqn, env, num_episodes=5):
    """
    :param dqn: DQN
    :param env: OpenAI gym environment
    :param num_episodes: number of evaluation episodes
    """
    episodes = [collect_episode(dqn, env, eps=0) 
                for _ in range(num_episodes)]
    data = Dataset(episodes)
    Q = dqn.Q(data.state, data.action)
    error = Q - data.discounted_return
    
    return {"Score": np.mean([ep.score for ep in episodes]),
            "Q-value": round(Q.mean().item(), 2),
            "Error": round(error.mean().item(), 2)}


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    parser.add_argument('--discount', type=float, default=0.98)
    parser.add_argument('--num_episodes', type=int, default=100)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.discount = args.discount
    
    dqn = DQN(env.observation_space.shape[0], env.action_space.n)
    
    data = Dataset([collect_episode(dqn, env, eps=1.)])
    episodes = []

    # Train online
    for i in range(args.num_episodes - 1):
        dqn.train(data)
        eps = 1 - i / args.num_episodes
        episode = collect_episode(dqn, env, eps=eps)
        data.add(episode)
        episodes.append(episode)
        if i % 10 == 0:
            print(f"Episode {i}: {evaluate_online(dqn, env)}")
            
    with open(f"data/dqn_{args.env_name}.pkl", "wb") as f:
        pickle.dump(data, f)
        
    # Train offline
    dqn = DQN(env.observation_space.shape[0], env.action_space.n)
    print(f"Before training offline: {evaluate_online(dqn, env)}")
    dqn.train(data, steps=5000)
    print(f"After training offline: {evaluate_online(dqn, env)}")
    
    # Evaluate offline 
    print(f"Offline evaluation: {evaluate_offline(dqn, data)}")