import numpy as np
import gymnasium as gym

env = gym.make("CartPole-v1")
#import pdb; pdb.set_trace()
obs, info = env.reset()

time_to_collapse = 0
ttc_arr = []
for _ in range(600):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    time_to_collapse += 1
    if terminated or truncated:
        ttc_arr.append(time_to_collapse)
        time_to_collapse = 0
        obs, info = env.reset()
env.close()
print(f"number of resets: {len(ttc_arr)}; average time to collapse: {np.mean(ttc_arr)}")