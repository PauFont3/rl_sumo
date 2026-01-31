import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.sumo_env import SumoEnv


def train():
   
    models_dir = os.path.join("models", "PPO")
    log_dir = "logs"
    checkpoint_dir = os.path.join(models_dir, "checkpoint_final")
  
    os.makedirs(checkpoint_dir, exist_ok=True) # Ensure checkpoint directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env = SumoEnv()  

    model_path = os.path.join(models_dir, "model_fase2_toro.zip")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading existing model from: {model_path}...")
    model = PPO.load(model_path, env=env, tensorboard_log=log_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000, 
        save_path=checkpoint_dir,
        name_prefix="fase3_mix"
    )

    model.learn(
        total_timesteps=3_000_000, 
        reset_num_timesteps=False,
        callback=checkpoint_callback,
        tb_log_name="PPO" # TensorBoard log name
    )

    final_path = os.path.join(models_dir, "model_fase3_final.zip")
    model.save(final_path)
    print(f"Training finished! Model saved at: {final_path}")

if __name__ == "__main__":
    train()