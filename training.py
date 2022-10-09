import gym
from network import dqn
from environment import experience, agent, wrappers
import torch
import torch.optim as optim  # Pytorch optimization package
import torch.nn as nn
import numpy as np
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0
MEAN_REWARD_BOUND = 19.0

gamma = 0.99
batch_size = 32
replay_size = 10000
learning_rate = 1e-4
sync_target_frames = 1000
replay_start_size = 10000

eps_start = 1.0
eps_decay = .999985
eps_min = 0.02
env = wrappers.make_env(DEFAULT_ENV_NAME)
device = torch.device("cuda")
replay_size = 10000
net = dqn.DQN(env.observation_space.shape, env.action_space.n).to(device)
target_net = dqn.DQN(env.observation_space.shape,
                     env.action_space.n).to(device)

buffer = experience.ExperienceReplay(replay_size)
agent = agent.Agent(env, buffer)

epsilon = eps_start

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
total_rewards = []
frame_idx = 0

best_mean_reward = None

while True:
    frame_idx += 1
    epsilon = max(epsilon*eps_decay, eps_min)

    reward = agent.play_step(net, epsilon, device=device)
    if reward is not None:
        total_rewards.append(reward)

        mean_reward = np.mean(total_rewards[-100:])

        print("%d:  %d games, mean reward %.3f, (epsilon %.2f)" % (
            frame_idx, len(total_rewards), mean_reward, epsilon))

        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best.dat")
            best_mean_reward = mean_reward
            if best_mean_reward is not None:
                print("Best mean reward updated %.3f" % (best_mean_reward))

        if mean_reward > MEAN_REWARD_BOUND:
            print("Solved in %d frames!" % frame_idx)
            break

    if len(buffer) < replay_start_size:
        continue

    batch = buffer.sample(batch_size)
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)

    next_state_values = target_net(next_states_v).max(1)[0]

    next_state_values[done_mask] = 0.0

    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v

    loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss_t.backward()
    optimizer.step()

    if frame_idx % sync_target_frames == 0:
        target_net.load_state_dict(net.state_dict())
