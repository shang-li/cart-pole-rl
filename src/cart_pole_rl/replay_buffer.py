"""
A replay buffer is a buffer of experiences that are used to train the agent. 
It is a circular buffer that stores the experiences and samples them randomly. 
Circular buffer stores data in continuous memory which makes sampling efficient.

Provide uniform random sampling and recency weighted sampling.
"""

import numpy as np

class ReplayBuffer:
    def __init__(self, capacity:int, obs_dim:int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.obs_next = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.int32)
        self.idx = -1
        self.size = 0
        self.priority = np.zeros(capacity, dtype=np.float32)
        self.forgetting_weight = 0.9

    def __len__(self):
        return self.size

    def append(self, obs:np.array, action:int, reward:float, obs_next:np.array, terminated:bool):
        self.size = min(self.size + 1, self.capacity)
        self.idx = (self.idx + 1) % self.capacity
        self.priority[self.idx] = 1.0
        self.priority = self.priority * self.forgetting_weight

        self.obs[self.idx, :] = obs
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.obs_next[self.idx, :] = obs_next
        self.terminated[self.idx] = terminated
    
    def uniform_sample(self, batch_size:int):
        idx = np.random.choice(self.size, batch_size, replace=True)
        return self.obs[idx], self.action[idx], self.reward[idx], self.obs_next[idx], self.terminated[idx]
    
    def recency_weighted_sample(self, batch_size:int):
        idx = np.random.choice(np.arange(self.capacity), batch_size, replace=False, p=self.priority / np.sum(self.priority))
        return self.obs[idx], self.action[idx], self.reward[idx], self.obs_next[idx], self.terminated[idx]

if __name__ == "__main__":
    rb = ReplayBuffer(2, 2)
    rb.append(np.array([1,1]), 1, 1.8, np.array([2,2]), False)
    rb.append(np.array([2,1]), 2, 1.8, np.array([2,2]), True)
    rb.append(np.array([3,1]), 3, 1.8, np.array([2,2]), True)
    rb.append(np.array([4,1]), 3, 1.8, np.array([2,2]), False)