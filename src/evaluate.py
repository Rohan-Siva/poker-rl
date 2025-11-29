import argparse
import rlcard
import torch
import numpy as np
import random
from src.agent import DQNAgent
from rlcard.agents import RandomAgent

def evaluate(args):
    env = rlcard.make('blackjack')
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load agent
    if isinstance(env.state_shape, list):
        state_shape = env.state_shape[0]
    else:
        state_shape = env.state_shape
    
    num_actions = env.num_actions
    agent = DQNAgent(num_actions, state_shape, device=args.device)
    agent.load(args.model_path)
    agent.epsilon = 0.0 # Greedy policy for evaluation

    # baseline is random moves
    random_agent = RandomAgent(num_actions=num_actions)

    print("Evaluating DQN Agent...")
    dqn_rewards = run_eval(env, agent, args.num_episodes)
    print(f"DQN Agent Average Reward: {np.mean(dqn_rewards):.4f}")
    print(f"DQN Agent Win Rate: {sum(r > 0 for r in dqn_rewards) / args.num_episodes:.4f}")

    print("\nEvaluating Random Agent...")
    random_rewards = run_eval(env, random_agent, args.num_episodes)
    print(f"Random Agent Average Reward: {np.mean(random_rewards):.4f}")
    print(f"Random Agent Win Rate: {sum(r > 0 for r in random_rewards) / args.num_episodes:.4f}")

def run_eval(env, agent, num_episodes):
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        while not env.is_over():
            if isinstance(agent, DQNAgent):
                action = agent.eval_step(state['obs'])
            else:
                action = agent.step(state)
            
            next_state, _ = env.step(action)
            state = next_state
        
        rewards.append(env.get_payoffs()[0])
    return rewards

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    evaluate(args)
