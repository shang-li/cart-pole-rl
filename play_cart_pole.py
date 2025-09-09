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

def policy(obs, model_path):
    net = PolicyNet(4, 16, 2)
    net.load_state_dict(torch.load(model_path))
    tau = 0.5
    probs = torch.softmax(net(torch.tensor(obs)) / tau, dim=0).detach().numpy()
    return np.random.choice([0, 1], p=probs)

time_to_collapse = 0
ttc_arr = []
for _ in range(600):
    #action = q_net_policy(obs)
    #action = policy(obs, "./data/crossent_policy.pth")
    action = policy(obs, "./data/REINFORCE_policy.pth")
    obs, reward, terminated, truncated, info = env.step(action)
    time_to_collapse += 1
    if terminated or truncated:
        ttc_arr.append(time_to_collapse)
        time_to_collapse = 0
        obs, info = env.reset()
env.close()
print(f"number of resets: {len(ttc_arr)}; average time to collapse: {np.mean(ttc_arr)}")
