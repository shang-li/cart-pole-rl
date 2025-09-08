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
    #action = q_net(torch.tensor(obs, dtype=torch.float32)).argmax().item()
    with torch.inference_mode():
        tau = 0.1
        logits = q_net(torch.tensor(obs, dtype=torch.float32))
        probs = torch.softmax(logits / tau, dim=0).cpu().numpy()
        action = np.random.choice([0, 1], p=probs)
    return action

def crossent_policy(obs):
    net = PolicyNet(4, 16)
    net.load_state_dict(torch.load("./data/crossent_policy.pth"))
    tau = 0.5
    val = torch.sigmoid(net(torch.tensor(obs)) / tau).item()
    return np.random.choice([0, 1], p=[1-val, val])

time_to_collapse = 0
ttc_arr = []
for _ in range(1000):
    #action = q_net_policy(obs)
    action = crossent_policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    time_to_collapse += 1
    if terminated or truncated:
        ttc_arr.append(time_to_collapse)
        time_to_collapse = 0
        obs, info = env.reset()
env.close()
print(f"number of resets: {len(ttc_arr)}; average time to collapse: {np.mean(ttc_arr)}")
