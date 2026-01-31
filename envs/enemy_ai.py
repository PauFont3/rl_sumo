import numpy as np
import math
import random


class EnemyStrategy:
    """
    Base class for enemy AI strategies.
    """
    def execute_strategy(self, enemy_pos, agent_pos):
        raise NotImplementedError("This method should be overridden by subclasses.")


class RandomEnemyStrategy(EnemyStrategy):
    """
    THE DRUNK 
    Strategy: Moves randomly.
    Just to add caos to the agent.
    """
    def execute_strategy(self, enemy_pos, agent_pos):
        # Generate a random movement vector
        angle = random.uniform(0, 2 * math.pi)
        force = 30.0
        
        fx = math.cos(angle) * force
        fy = math.sin(angle) * force

        return fx, fy


class AggressiveEnemyStrategy(EnemyStrategy):
    """
    THE BULL
    Strategy: Moves towards the agent to collide with it.
    """
    def execute_strategy(self, enemy_pos, agent_pos):
        dx = agent_pos[0] - enemy_pos[0]
        dy = agent_pos[1] - enemy_pos[1]
        
        # Exact direction to the agent
        angle = math.atan2(dy, dx)
        
        # Apply a force towards the agent
        force = 40.0
        fx = math.cos(angle) * force
        fy = math.sin(angle) * force
        
        return fx, fy
    

class DefensiveEnemyStrategy(EnemyStrategy):
    """
    The ROCK
    Strategy: Controls the center.
    """
    def execute_strategy(self, enemy_pos, agent_pos):
        dist = np.linalg.norm(agent_pos - enemy_pos)

        # Objective: Control the center of the arena
        target_x = 0.0
        target_y = 0.0
        
        # If the agent is too close, attack the agent 
        if dist < 1.5:
            target_x = agent_pos[0]
            target_y = agent_pos[1]

        # Calculate direction to the target position
        dx = target_x - enemy_pos[0]
        dy = target_y - enemy_pos[1]
        angle = math.atan2(dy, dx)

        # Apply a force towards the target position
        force = 30.0 
        fx = math.cos(angle) * force
        fy = math.sin(angle) * force
        
        return fx, fy