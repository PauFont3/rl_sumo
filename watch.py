import gymnasium as gym
from stable_baselines3 import PPO
from envs.sumo_env import SumoEnv
import time

def main():
    env = SumoEnv()
    env.reset()
    print("[INFO] Environment initialized")
    model_path = "models/PPO/model_fase3_final.zip" 
    
    print(f"\n[INFO] Loading model: {model_path}")
    

    try:
        model = PPO.load(model_path, env=env)
        print("[INFO] Model loaded successfully")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    episodes = 10
    for ep in range(episodes): 
        print(f"--- Episode {ep + 1} ---")
        obs, _ = env.reset()
        done = False

        step_count = 0
        total_reward = 0.0
       
        while not done:
            step_count += 1
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            env.render()

        print(f"  Steps: {step_count}")
        print(f"  Total reward: {total_reward:.2f}")         

    print("[INFO] Visualization completed")   
    env.close()


if __name__ == "__main__":
    main()