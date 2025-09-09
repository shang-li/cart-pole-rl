import os
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from cart_pole_rl.policy_net import PolicyNet


class REINFORCEAgent():
    NSTEPS = 3000
    EPISODES_PER_STEP = 100
    def __init__(self, env):
        self.env = env
        self.policy_net = PolicyNet(4, 16, 2)
        self.writer = SummaryWriter(log_dir="runs/reinforce1")
        self.gamma = 0.99
    
    def act(self, obs):
        act_probs = F.softmax(self.policy_net(torch.tensor(obs, dtype=torch.float32)), dim=0).detach().numpy()
        return np.random.choice([0, 1], p=act_probs)
    
    def discount_cusum(self, r, gamma):
        r = np.asarray(r, dtype=np.float32)
        out = np.zeros_like(r, dtype=np.float32)
        running = 0.0
        for t in range(len(r) - 1, -1, -1):
            running = r[t] + gamma * running
            out[t] = running
        return out

    def collect_data(self):
        is_good = False
        obs, _ = self.env.reset()
        X_candidates = []
        y_candidates = []
        rewards = []
        episode_returns = []
        for _ in range(self.EPISODES_PER_STEP):
            rw_list = []
            obs_list = []
            action_list = []
            done = False
            while not done:
                action = self.act(obs)
                obs_next, reward, terminated, truncated, _ = self.env.step(action)
                obs_list.append(obs)
                action_list.append(action)
                rw_list.append(reward)
                done = truncated or terminated
                obs = obs_next
                if done:
                    X_candidates.append(np.vstack(obs_list))
                    y_candidates.append(np.array(action_list))
                    rewards.append(self.discount_cusum(rw_list, self.gamma))
                    episode_returns.append(np.sum(rw_list))
                    obs, _ = self.env.reset()
        # Use average episode return as the solved signal
        if len(episode_returns) > 0 and np.mean(episode_returns) >= 300:
            is_good = True
        return np.vstack(X_candidates), np.hstack(y_candidates), np.hstack(rewards), is_good
        
    def train(self):
        # new pi = argmin -Q(a, s) log pi(a | obs)
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-3)
        for i in tqdm(range(self.NSTEPS)):
            X, y, rewards, is_good = self.collect_data()
            rewards_normed = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            if not is_good:
                optimizer.zero_grad()
                X_t = torch.tensor(X, dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.long)
                logits = self.policy_net(X_t)
                # REINFORCE loss: -E[return * log pi(a|s)]
                log_probs = F.log_softmax(logits, dim=1).gather(1, y_t.view(-1, 1)).squeeze(1)
                returns_t = torch.tensor(rewards_normed, dtype=torch.float32)
                loss = -(returns_t * log_probs).mean()
                loss.backward()
                optimizer.step()

                self.writer.add_scalar("reward/step", np.mean(rewards), i)
                self.writer.add_scalar("loss/policy", loss.item(), i)
                if i % 50 == 0:
                    for name, p in self.policy_net.named_parameters():
                        self.writer.add_histogram(f"params/{name}", p.data, i)
                        if p.grad is not None:
                            self.writer.add_histogram(f"grads/{name}", p.grad, i)
            else:
                print("problem solved!")
                break

        ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        DATA_DIR = os.path.join(ROOT, "data")

        # Make sure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        # Save model
        model_path = os.path.join(DATA_DIR, "REINFORCE_policy.pth")
        torch.save(self.policy_net.state_dict(), model_path)
        print("Successfully trained a model and stored!")


def make_env():
    return gym.make("CartPole-v1")

env = make_env()
agent = REINFORCEAgent(env=env)
agent.train()
