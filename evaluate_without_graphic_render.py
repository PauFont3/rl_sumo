from stable_baselines3 import PPO
from envs.sumo_env import SumoEnv
import os
from tqdm import tqdm # Progress bar (pip install tqdm)

def evaluate():
    env = SumoEnv() 
    model_path = os.path.join("models", "PPO", "model_fase3_final.zip")
    model = PPO.load(model_path, env=env)

    n_games = 1000
    wins = 0
    
    print(f" Simulating {n_games} games without graphic rendering...")

    for _ in tqdm(range(n_games)):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        if episode_reward > 50: # Adjust threshold as needed
            wins += 1

    win_rate = (wins / n_games) * 100
    print(f"\n RESULTS:")
    print(f"   Victories: {wins}")
    print(f"   Defeats:  {n_games - wins}")
    print(f"   Win Rate:  {win_rate:.2f}%")

    if win_rate > 85:
        print(" Score: SUMO BEAST")
    elif win_rate > 70:
        print(" Score: PROFESSIONAL FIGHTER")
    else:
        print(" Score: AMATEUR (Needs more training)")

if __name__ == "__main__":
    evaluate()