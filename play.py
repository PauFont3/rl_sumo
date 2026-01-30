import time 
import numpy as np
import pygame

from envs.sumo_env import SumoEnv

def main():
    env = SumoEnv()
    obs, _ = env.reset()
    env.render()

    print("Starting simulation...")

    step = 0
    running = True

    while running and step < 500:
        step += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = np.array([0.9, 0.2])

        # Send actions to environment
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        if done:
            print(" --- RESETTING ENVIRONMENT --- ")
            obs, _ = env.reset()

    print("Simulation finished")
    env.close()


if __name__ == "__main__":
    main()