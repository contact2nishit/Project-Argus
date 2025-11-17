import sys
sys.path.append('..')
from src.gnn_agent import Agent, obs2graph
from environment.geospatial_env import Env
from shapely.geometry import box # type: ignore
from pettingzoo.test import parallel_api_test #type: ignore

import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import torch # type: ignore
import torch.nn.functional as F # type: ignore

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment setup
    north, south, east, west = 35.6873, 35.6864, 139.764, 139.7629
    origin = (west, south)
    bbox = box(west, south, east, north)

    num_agents = 5
    env = Env(origin, bbox, num_agents=num_agents, battery_capacity=500)

    agent = env.agents[0]
    obs_dim = env.observation_space(agent).shape[0]
    act_dim = env.action_space(agent).shape[0]

    # Model setup
    model = Agent(
        num_agents=num_agents,
        in_dim=obs_dim*5,
        hidden_dim=64,
        out_dim=act_dim
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)

    # Training hyperparameters
    max_episodes = 500
    max_steps = 200
    gamma = 0.95
    value_coef = 0.1
    entropy_coef = 0.01

    # Logging
    log_data = []

    for ep in range(max_episodes):
        observations, infos = env.reset()
        
        ep_rewards = []

        saved_log_probs = []
        saved_values = []
        saved_rewards = []
        saved_entropies = []

        for t in range(max_steps):
            # Convert observations dict/list → graph
            data = obs2graph(observations).to(device)

            # Forward pass
            means, values = model(data)  # [num_agents, act_dim], [num_agents]
            log_std = model.policy.log_std
            std = torch.exp(log_std).unsqueeze(0).expand_as(means)
            dist = torch.distributions.Normal(means, std)

            # Sample actions
            sampled_actions = dist.rsample()  # rsample() allows gradients
            low = torch.tensor(env.action_space(agent).low, dtype=torch.float32).to(device)
            high = torch.tensor(env.action_space(agent).high, dtype=torch.float32).to(device)
            sampled_actions = torch.clamp(sampled_actions, min=low, max=high)

            log_prob = dist.log_prob(sampled_actions).sum(dim=-1)  # [num_agents]
            entropy = dist.entropy().sum(dim=-1)  # [num_agents]

            actions_dict = {agent: sampled_actions[i].detach().cpu().numpy() 
                for i, agent in enumerate(env.agents)}

            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions_dict)

            # Convert reward dict → tensor with correct agent ordering
            reward_tensor = torch.tensor([rewards[a] for a in sorted(rewards.keys())], dtype=torch.float32, device=device)
            reward_tensor = reward_tensor * 10.0 
            reward_tensor = torch.clamp(reward_tensor, -10.0, 10.0)

            # Store trajectory
            saved_log_probs.append(log_prob)
            saved_values.append(values)
            saved_rewards.append(reward_tensor)
            saved_entropies.append(entropy)

            ep_rewards.append(reward_tensor.mean().item())

            # Terminate if all agents done
            if any(list(terminations.values())) or all(list(truncations.values())):
                break

        # Stack trajectory tensors: [T, num_agents]
        logp_tensor = torch.stack(saved_log_probs)
        values_tensor = torch.stack(saved_values)
        rewards_tensor = torch.stack(saved_rewards)
        entropies_tensor = torch.stack(saved_entropies)

        T, N = rewards_tensor.shape

        # Compute discounted returns
        
        returns = torch.zeros_like(rewards_tensor)
        future = torch.zeros(N, device=device)
        for i in reversed(range(T)):
            future = rewards_tensor[i] + gamma * future
            returns[i] = future

        returns_mean = returns.mean()
        returns_std = returns.std(unbiased=False) + 1e-8
        returns = (returns - returns_mean) / returns_std
        # Compute advantages
        advantages = returns - values_tensor
        adv_flat = advantages.view(-1)
        advantages = (advantages - adv_flat.mean()) / (adv_flat.std(unbiased=False) + 1e-8)

        # Losses
        policy_loss = -(logp_tensor * advantages.detach()).mean()
        value_loss = F.mse_loss(values_tensor, returns)
        entropy_loss = entropies_tensor.mean()
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

        # Backprop
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        opt.step()

        avg_ep_reward = sum(ep_rewards) / max(1, len(ep_rewards))
        print(f"Episode {ep:04d} | Steps {T} | AvgReward {avg_ep_reward:.3f} | "
              f"policy {policy_loss.item():.4f} | value {value_loss.item():.4f} | entropy {entropy_loss.item():.4f}")

        log_data.append({'t': T, 'r': avg_ep_reward})

    # Save logs to DataFrame
    df = pd.DataFrame(log_data)
    plot(df)

def plot(df):
    df['cum_t'] = df['t'].cumsum()
    plt.plot(df['cum_t'], df['r'], label='AvgReward')
    plt.xlabel('Total Steps')
    plt.ylabel('Avg Reward')
    plt.title('Reward vs Total Steps')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run()
