import gym
import os
import numpy as np
from gym.wrappers import RecordVideo

class CartPoleEnv:
    def __init__(self, video_dir=None):
        self.env = gym.make('CartPole-v1', render_mode='rgb_array')
        if video_dir:
            self.env = RecordVideo(self.env, video_dir, episode_trigger=lambda e: True)
        self.state = self.env.reset()

    def reset(self):
        self.state = self.env.reset()
        return self.state

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            next_state, reward, done, truncated, info = result
            done = done or truncated
        self.state = next_state
        return next_state, reward, done, info

    def close(self):
        self.env.close()

if __name__ == "__main__":
    video_dir = './episodes'
    os.makedirs(video_dir, exist_ok=True)
    
    env = CartPoleEnv(video_dir=video_dir)
    state = env.reset()
    done = False

    while not done:
        action = env.env.action_space.sample()  # Random action
        next_state, reward, done, info = env.step(action)

    env.close()