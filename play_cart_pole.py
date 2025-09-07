import numpy as np
import gymnasium as gym
import torch
from cart_pole_rl.dqn import DQN
from cart_pole_rl.policy_net import PolicyNet

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()


def q_net_policy(obs):
    q_net = DQN(4, 16, 2)
    q_net.load_state_dict(torch.load("./data/q_net.pth"))
    action = q_net(torch.tensor(obs, dtype=torch.float32)).argmax().item()
    return action

def crossent_policy(obs):
    net = PolicyNet(4, 16)
    net.load_state_dict(torch.load("./data/crossent_policy.pth"))
    val = torch.sigmoid(net(torch.tensor(obs)))
    return int(val >=0.5 )

time_to_collapse = 0
ttc_arr = []
for _ in range(3000):
    #action = env.action_space.sample()
    action = crossent_policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    time_to_collapse += 1
    if terminated or truncated:
        ttc_arr.append(time_to_collapse)
        time_to_collapse = 0
        obs, info = env.reset()
env.close()
print(f"number of resets: {len(ttc_arr)}; average time to collapse: {np.mean(ttc_arr)}")
