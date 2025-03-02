# DQN main.py

import numpy as np
import torch
import cartpole
import os

def main():
    # Params
    max_episodes = 5

    print("Hello World!")
    print("PyTorch version: ", torch.__version__)
    print("CUDA available: ", torch.cuda.is_available())
    print("NumPy version: ", np.__version__)

    # Create a directory to save the videos
    video_dir = './episodes'
    os.makedirs(video_dir, exist_ok=True)

    env = cartpole.CartPoleEnv(video_dir=video_dir)
    state = env.reset()
    done = False

    for i in range(max_episodes):
        env.reset()
        done = False
        while not done:
            action = env.env.action_space.sample()
            next_state, reward, done, info = env.step(action)

    env.close()

if __name__ == "__main__":
    main()