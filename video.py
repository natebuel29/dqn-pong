from wrappers import wrappers
import gym
from network.dqn import DQN
import torch
import time
import numpy as np
FPS = 25
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
model = 'PongNoFrameskip-v4-best.dat'
record_folder = "video"
visualize = True

env = wrappers.make_env(DEFAULT_ENV_NAME)
if record_folder:
    env = gym.wrappers.Monitor(env, record_folder, force=True)
net = DQN(env.observation_space.shape, env.action_space.n)
net.load_state_dict(torch.load(
    model, map_location=lambda storage, loc: storage))

state = env.reset()
total_reward = 0.0

while True:
    start_ts = time.time()
    # if visualize:
    #     env.render()
    state_v = torch.tensor(np.array([state], copy=False))
    q_vals = net(state_v).data.numpy()[0]
    action = np.argmax(q_vals)

    state, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        break
    if visualize:
        delta = 1/FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
print("Total reward: %.2f" % total_reward)

if record_folder:
    env.close()
