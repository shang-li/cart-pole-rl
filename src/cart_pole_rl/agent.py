# An agent hosts 1) interacting with env 2) act based on policy 3) collect learning from collected data 
# what'll be deployed is 2); 1) is done by passing env class and expose play API; 3) is done by passing
# a replay buffer, where data is sampled and used for training: forward and backward and gradient descent
import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
from cart_pole_rl.dqn import DQN
from cart_pole_rl.replay_buffer import ReplayBuffer


class DQN_Agent():
    def __init__(self, env):
        self.env = env
        self.q_net = DQN(4, 8, 2)
        self.q_target_net = DQN(4, 8, 2)
        self.q_target_net.load_state_dict(self.q_net.state_dict())
        self.replay_buffer = ReplayBuffer(capacity=15000, obs_dim=4)
        self.episode_steps_log = []
        self.action_log = []
        self.rng = np.random.default_rng(seed=66)

    def act_epsilon_greedy(self, obs, epsilon=0.5):
        rn_val = self.rng.uniform(0, 1)
        if rn_val < epsilon:
            action = int(np.random.choice([0, 1], p=[0.5, 0.5]))
        else:
            q_val = self.q_net(torch.tensor(obs))
            action = torch.argmax(q_val).item()
        self.action_log.append(action)
        return action

    def run(self, is_training=False, batch_size=100, total_steps=300, start_learning=1000, target_update=1000, eps_start=1.0, eps_end=0.05, eps_decay=20000):
        obs, _ = self.env.reset()
        episode_steps = 0
        if is_training:
            optimizer = torch.optim.Adam(self.q_net.parameters(), lr=5e-2)

        for i in range(total_steps):
            episode_steps += 1

            eps = eps_end + (eps_start - eps_end) * max(0.0, (eps_decay - i) / eps_decay)
            action = self.act_epsilon_greedy(obs, epsilon=eps)
            obs_next, reward, terminated, truncated, _ = self.env.step(action)
            if is_training:
                self.replay_buffer.append(obs, action, reward, obs_next, terminated or truncated)
                
                if i > start_learning and (i % batch_size // 100) == 0:
                    # train
                    #print("update weights......")
                    optimizer.zero_grad()
                    ob, ac, rw, ob2, dn = self.replay_buffer.uniform_sample(batch_size)
                    # to torch (correct dtypes/shapes/devices)
                    ob_t  = torch.tensor(ob,  dtype=torch.float32)           # [B,4]
                    ac_t  = torch.tensor(ac,  dtype=torch.long).unsqueeze(1)  # [B,1]
                    rw_t  = torch.tensor(rw,  dtype=torch.float32)           # [B]
                    ob2_t = torch.tensor(ob2, dtype=torch.float32)           # [B,4]
                    dn_t  = torch.tensor(dn,  dtype=torch.float32) 

                    with torch.no_grad():
                        next_q = self.q_target_net(ob2_t).max(1).values                     # [B]
                        q_target = rw_t + 0.99 * (1.0 - dn_t) * next_q
                    
                    q_pred = self.q_net(ob_t).gather(1, ac_t).squeeze(1)
                    #q_pred = self.q_net(torch.tensor(obs_batch)).gather(1, torch.tensor(action_batch.reshape(-1, 1))).squeeze()
                    loss = F.smooth_l1_loss(q_pred, q_target)
                    loss.backward()
                    optimizer.step()
                
                # update target network
                if i % target_update == 0:
                    self.q_target_net.load_state_dict(self.q_net.state_dict())

            if terminated or truncated:
                self.episode_steps_log.append(episode_steps)
                episode_steps = 0
                obs, _ = self.env.reset()
            else:
                obs = obs_next

if __name__ == "__main__":
    #env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CartPole-v1")
    dqn_agent = DQN_Agent(env)
    obs, info = env.reset()
    dqn_agent.run(batch_size=600, total_steps=30000, is_training=True)
    print(dqn_agent.action_log)
    import matplotlib.pyplot as plt
    plt.plot(dqn_agent.episode_steps_log)
    plt.show()
            



