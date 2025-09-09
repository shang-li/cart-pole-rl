import os
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
from cart_pole_rl.policy_net import PolicyNet


class CrossEntAgent():
    EPISODES_PER_TRAIN = 5000
    OUTER_ITERS = 10
    INNER_ITERS = 30
    def __init__(self, env, top_pct=0.7):
        self.env = env
        self.policy_net = PolicyNet(4, 16, 2)
        self.rng = np.random.default_rng(seed=66)
        self.top_pct = top_pct

    def act(self, obs):
        act_probs = F.softmax(self.policy_net(torch.tensor(obs)), dim=0).detach().numpy()
        return np.random.choice([0, 1], p=act_probs)
    
    def train(self):
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-2)
        rw_log = []
        for i in range(self.OUTER_ITERS):
            print(f"collect {i}th round of data...")
            X, y, is_good, avg_rw = self.collect_data()
            print(f"new dataset has average reward: {avg_rw}")
            rw_log.append(avg_rw)
            if is_good:
                print("Stop training due to good performance...")
            else:
                print(f"start training with {i + 1}th dataset")
                for j in tqdm(range(self.INNER_ITERS), desc="Steps"):
                    optimizer.zero_grad()
                    out = self.policy_net(torch.tensor(X))
                    criterion = torch.nn.CrossEntropyLoss()
                    loss = criterion(out, torch.LongTensor(y))
                    loss.backward()
                    optimizer.step()
        ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        DATA_DIR = os.path.join(ROOT, "data")

        # Make sure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        # Save model
        model_path = os.path.join(DATA_DIR, "crossent_policy.pth")
        torch.save(self.policy_net.state_dict(), model_path)
        print("Successfully trained a model and stored!")
        
    def collect_data(self):
        is_good = False
        obs, _ = self.env.reset()
        X_candidates = []
        y_candidates = []
        rewards = []
        for _ in range(self.EPISODES_PER_TRAIN):
            total_rw = 0
            obs_list = []
            action_list = []
            done = False
            while not done:
                action = self.act(obs)
                obs_next, reward, terminated, truncated, _ = self.env.step(action)
                obs_list.append(obs)
                action_list.append(action)
                total_rw += reward
                done = truncated or terminated
                obs = obs_next
                if done:
                    X_candidates.append(np.vstack(obs_list))
                    y_candidates.append(np.stack(action_list))
                    rewards.append(total_rw)
                    obs, _ = self.env.reset()
        if np.mean(rewards) >= 300:
            is_good = True
        val = np.quantile(rewards, self.top_pct)
        X = np.vstack([x for x, rw in zip(X_candidates, rewards) if rw >= val])
        y = np.hstack([y for y, rw in zip(y_candidates, rewards) if rw >= val])
        return X, y, is_good, np.mean(rewards)

    
def make_env():
    return gym.make("CartPole-v1")

env = make_env()
agent = CrossEntAgent(env=env)
agent.train()
