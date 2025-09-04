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
        self.q_net = DQN(4, 16, 2)
        self.q_target_net = DQN(4, 16, 2)
        self.replay_buffer = ReplayBuffer(capacity=10000, obs_dim=4)
        self.episode_steps_log = []
        self.action_log = []
        self.rng = np.random.default_rng(seed=66)

    def act_epsilon_greedy(self, obs, epsilon=0.5):
        rn_val = self.rng.uniform(0, 1)
        if rn_val < epsilon:
            action = int(np.random.choice([0, 1], p=[0.5, 0.5]))
        else:
            action_prob = self.q_net(torch.tensor(obs))
            action = torch.argmax(action_prob).item()
        self.action_log.append(action)
        return action

    def run(self, is_training=False, batch_size=400, total_steps=30000):
        obs, _ = self.env.reset()
        episode_steps = 0
        if is_training:
            optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)

        for i in range(total_steps):
            episode_steps += 1
            #nbatch = i // (batch_size * 3)
            action = self.act_epsilon_greedy(obs, epsilon=0.1)
            obs_next, reward, terminated, truncated, _ = self.env.step(action)
            if is_training:
                self.replay_buffer.append(obs, action, reward, obs_next, terminated or truncated)
            
                if i > 0 and i % (batch_size//10) == 0:
                    # train
                    optimizer.zero_grad()
                    obs_batch, action_batch, reward_batch, obs_next_batch, terminated_batch = self.replay_buffer.uniform_sample(batch_size)
    
                    q_target = torch.tensor(
                        reward_batch + 0.8 * terminated_batch * np.max(self.q_target_net(torch.tensor(obs_next_batch)).detach().numpy(), axis=1)
                    )
                    q_pred = self.q_net(torch.tensor(obs_batch)).gather(1, torch.tensor(action_batch.reshape(-1, 1)))
                    loss = F.smooth_l1_loss(q_pred, q_target)
                    loss.backward()
                    optimizer.step()
                
                # update target network
                if i % batch_size == 0:
                    self.q_target_net.load_state_dict(self.q_net.state_dict())

            if terminated or truncated:
                self.episode_steps_log.append(episode_steps)
                episode_steps = 0
                obs, _ = self.env.reset()
            obs = obs_next

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    dqn_agent = DQN_Agent(env)
    obs, info = env.reset()
    dqn_agent.run(batch_size=10, total_steps=300, is_training=True)
    print(dqn_agent.action_log)
    import matplotlib.pyplot as plt
    plt.plot(dqn_agent.episode_steps_log)
    plt.show()
            



