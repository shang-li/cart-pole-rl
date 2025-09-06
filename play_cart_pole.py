import numpy as np
import gymnasium as gym
import torch
from cart_pole_rl.dqn import DQN

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()


def policy(obs):
    q_net = DQN(4, 16, 2)
    q_net.load_state_dict(torch.load("./data/q_net.pth"))
    action = q_net(torch.tensor(obs, dtype=torch.float32)).argmax().item()
    return action

time_to_collapse = 0
ttc_arr = []
for _ in range(5000):
    #action = env.action_space.sample()
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    time_to_collapse += 1
    if terminated or truncated:
        ttc_arr.append(time_to_collapse)
        time_to_collapse = 0
        obs, info = env.reset()
env.close()
print(f"number of resets: {len(ttc_arr)}; average time to collapse: {np.mean(ttc_arr)}")
