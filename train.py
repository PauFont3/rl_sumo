import gymnasium as gym
from stable_baselines3 import PPO
from envs.sumo_env import SumoEnv
import os


def main():
    models_dir = "models/PPO"
    log_dir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create Environment
    env = SumoEnv()

    print("--- LOADING PPO ALGORITHM ---")
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir)

    print("--- STARTING TRAINING ---")
    print("The robot will start training. This may take a while...")
    
    TIMESTEPS = 1_000_000
    for i in range(1, 11): # 10 x 1_000_000 = 10_000_000 total steps
        print(f"\nITERATION {i}/10")
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name="PPO"
        )
        
        # Save the model
        save_path = f"{models_dir}/{TIMESTEPS*i}"
        model.save(save_path)
        print(f"--> Model saved at: {save_path}")

    env.close()
    print("\n" + "="*60)
    print("--- TRAINING COMPLETED ---")
    print("="*60)
    print(f"\n Total steps trained: {TIMESTEPS*10:,}")
    print(f" Models saved in: {models_dir}/")
    print(f" Logs available in: {log_dir}/")
    print("\nTo visualize progress:")
    print("  tensorboard --logdir=logs")
    print("\nTo watch the agent play:")
    print("  python watch.py")


if __name__ == "__main__":
    main()