# SumoBot-RL
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Lib](https://img.shields.io/badge/Library-Stable__Baselines3-orange)
![Sim](https://img.shields.io/badge/Simulation-PyBullet-green)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen)

> **Autonomous RL agent trained for Sumo Robotics competitions, capable of adapting its strategy against aggressive, defensive, and stochastic opponents.**

This project implements a Reinforcement Learning agent using **PPO (Proximal Policy Optimization)**. The agent's goal is to push the opponent out of the *Dohyo* (ring) while remaining inside, using data from simulated sensors (LiDAR/Distance and relative positioning).

### The Brain
* **Observation Space:**
    * Self Position (x, y).
    * Enemy Position (x, y).
    * Yaw (Orientation).
    * Distance to edge (Survival instinct).
* **Action Space:** Continuous velocity control (Left/Right Motors).

### The Rivals (Strategy Pattern)
To prevent *overfitting* (the agent only learning to beat one specific move), the **Strategy Design Pattern** was implemented.
| Strategy | Behavior | Training Goal |
| :--- | :--- | :--- |
| **Aggressive** | Charges directly at the agent. | Learning to dodge and counter-attack. |
| **Defensive** | Stays in the center and rotates. | Learning to generate pushing force. |
| **Random** | Unpredictable stochastic movement. | Generalization and robustness. |
