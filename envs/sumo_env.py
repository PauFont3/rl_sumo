import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
import pygame

from physics.mapper import convert_given_points_to_robot_parameters
from envs.enemy_ai import RandomEnemyStrategy, AggressiveEnemyStrategy, DefensiveEnemyStrategy

# ----------------------------
# --- SIMULATION CONSTANTS ---
# ----------------------------
DT = 0.1    # Delta time (Time interval per step)
INITIAL_RING_RADIUS = 5.0   
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 900


# -------------------------------
# --- BATTLE ROYALE CONSTANTS ---
# -------------------------------
SHRINK_RATE = 0.01      # Rate at which the ring shrinks per step
SHRINK_START_STEP = 200 # Step at which the ring starts shrinking


class SumoEnv(gym.Env):
    """
    Gymnasium environment for a Sumo Robot Battle.

    The agent controls a robot in a sumo ring, aiming to push an enemy robot out of the ring
    while avoiding being pushed out itself. The ring shrinks over time to increase difficulty.
    The agent's robot design is randomly generated at the start of each episode.
    """

    def __init__(self):
        """
        Initialize the Sumo environment.
        - Physics parameters
        - Initial robot positions
        - Action and observation spaces
        """
        super(SumoEnv, self).__init__()

        # Rendering variables
        self.screen = None
        self.clock = None

        # Physics parameters
        self.dt = DT
        self.ring_radius = INITIAL_RING_RADIUS
        self.current_step_count = 0
  
        # --- AGENT ROBOT ---
        self.agent_designed = None
        self.agent_position = np.array([-2.0, 0.0], dtype=np.float32)
        self.agent_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.agent_orientation = 0.0

        # --- ENEMY ROBOT ---
        self.enemy_position = np.array([2.0, 0.0], dtype=np.float32)
        self.enemy_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.enemy_mass = 5.0 
        self.strategy = None
        
        # --- Action Space ---
        # - Continuous actions: [acceleration, direction]
        # - acceleration: -1.0 (backwards) to 1.0 (forwards)
        # - direction: -1.0 (left) to 1.0 (right)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- Observation Space ---
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32),
            "velocity": spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32),
            "ring_radius": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            "dist_to_center": spaces.Box(low=0.0, high=15.0, shape=(1,), dtype=np.float32),
            "orientation": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "enemy_position": spaces.Box(low=-20.0, high=20.0, shape=(2,), dtype=np.float32)
        })


    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        - Resets the ring radius and step counter.
        - Generates a new random robot design for the agent.
        - Resets positions and velocities of both robots.
        """
        super().reset(seed=seed)

        self.ring_radius = INITIAL_RING_RADIUS
        self.current_step_count = 0 # Restart the steps counter

        # --- GENERATE RANDOM ROBOT ---
        # Every time an episode starts, we test a different design
        random_design = [random.randint(0, 4) for _ in range(5)]
        self.agent_designed = convert_given_points_to_robot_parameters(random_design)

        # --- RESET AGENT PHYSICS ---
        self.agent_position = np.array([-2.0, 0.0], dtype=np.float32)
        self.agent_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.agent_orientation = random.uniform(0, 2 * math.pi) # Random initial orientation

        # --- SELECT ENEMY STRATEGY ---
        available_strategies = [
            RandomEnemyStrategy(), 
            AggressiveEnemyStrategy(), 
            DefensiveEnemyStrategy()
        ]
        self.strategy = random.choice(available_strategies)

        # --- RESET ENEMY PHYSICS ---
        self.enemy_position = np.array([2.0, 0.0], dtype=np.float32)
        self.enemy_velocity = np.array([0.0, 0.0], dtype=np.float32)


        return self._get_obs(), {}        


    def step(self, action):
        """
        Executes one time step within the simulation.
        - Updates the ring size if necessary.
        - Applies the agent's action to update its physics.
        - Updates the enemy robot's physics.
        - Checks for collisions between robots.
        - Calculates the reward and checks for termination conditions.
        """
        self.current_step_count += 1

        self.update_ring_size()

        self.update_agents_physics(action)
        self.update_enemy_physics()

        self.handle_collisions()

        self.apply_movement_and_friction()

        reward, terminated, truncated = self.compute_rewards_and_termination()
    
        return self._get_obs(), reward, terminated, truncated, {}


    def _get_obs(self):
        """
        Get the current observation of the environment.
        - Positions (x,y) of the agent and enemy.
        - Agent's velocity.
        - Current ring radius.
        - Distance to center and orientation of the agent.
        - Agent's orientation as (cos, sin) for continuity.
        """
        # Euclidean distance to center
        dist = math.sqrt(self.agent_position[0]**2 + self.agent_position[1]**2)

        return {
            "position": np.array(self.agent_position, dtype=np.float32),
            "enemy_position": np.array(self.enemy_position, dtype=np.float32),
            "velocity": np.array(self.agent_velocity, dtype=np.float32),
            "ring_radius": np.array([self.ring_radius], dtype=np.float32),
            "dist_to_center": np.array([dist], dtype=np.float32),
            "orientation": np.array([math.cos(self.agent_orientation), math.sin(self.agent_orientation)], dtype=np.float32)
        }


    def render(self):
        """
        Render the current state of the environment using Pygame.
        Draws:
            - The sumo ring with borders
            - Agent robot (blue/gray)
            - Enemy robot (red)
            - Relevant information (HUD)
        """
        # Initialize Pygame if not already done
        if (self.screen is None):
            pygame.init()
            pygame.display.set_caption("RL Sumo Simulation")
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        
        # Background
        self.screen.fill((50, 40, 30)) # White

        # Camera zoom configuration
        SCALE = 60 
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 # Screen center
        
        # --- DRAWING THE RING ---
        ring = int(self.ring_radius * SCALE)
        pygame.draw.circle(self.screen, (230, 200, 150), (cx, cy), ring) # Sand
        pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), ring, 3) # White border

        # Initial sumo lines in the center (shikiri-sen)
        pygame.draw.line(self.screen, (160, 140, 90), (cx - 120, cy - 30), (cx - 120, cy + 30), 5)
        pygame.draw.line(self.screen, (160, 140, 90), (cx + 120, cy - 30), (cx + 120, cy + 30), 5)

        # --- DRAWING THE AGENT ---
        agent_x = int(cx + self.agent_position[0] * SCALE)
        agent_y = int(cy + self.agent_position[1] * SCALE)

        # Robot body (Gray with blue borders)
        pygame.draw.circle(self.screen, (50, 100, 200), (agent_x, agent_y), 25)
        pygame.draw.circle(self.screen, (200, 200, 200), (agent_x, agent_y), 20)

        # Direction Line, to indicate orientation
        dir_x = agent_x + math.cos(self.agent_orientation) * 25
        dir_y = agent_y + math.sin(self.agent_orientation) * 25
        pygame.draw.line(self.screen, (255, 0, 0), (agent_x, agent_y), (dir_x, dir_y), 3)

        # --- DRAWING THE ENEMY ---
        enemy_x = int(cx + self.enemy_position[0] * SCALE)
        enemy_y = int(cy + self.enemy_position[1] * SCALE)

        # Enemy body
        pygame.draw.circle(self.screen, (255, 50, 50), (enemy_x, enemy_y), 25) 
        pygame.draw.circle(self.screen, (100, 0, 0), (enemy_x, enemy_y), 25, 2) # Dark border

        # Draw the line between the agent and the enemy
        pygame.draw.line(self.screen, (0, 255, 0), (agent_x, agent_y), (enemy_x, enemy_y), 1) 

        # Information text (HUD)
        font = pygame.font.SysFont("Arial", 18)
        info_text = f"Radio Ring: {self.ring_radius:.2f}m | Step: {self.current_step_count}"
        text_surf = font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (10, 10))

        # Refresh screen
        pygame.display.flip()
        self.clock.tick(30) # Limit to 30 FPS



    # -----------------------------
    # --- AUXILIARY FUNCTIONS -----
    # -----------------------------
    
    def update_ring_size(self):
        if (self.current_step_count >= SHRINK_START_STEP):
            self.ring_radius -= SHRINK_RATE
            self.radius = max(1.0, self.ring_radius) # Minimum radius

    
    def update_agents_physics(self, action):
        # Limit actions for safety
        throttle = np.clip(action[0], -1.0, 1.0) # Forward / Backward
        steering = np.clip(action[1], -1.0, 1.0) # Left / Right

        if self.agent_designed is None:
            mass, force, mu, agility = 5.0, 50.0, 0.5, 1.0
        else:
            mass = self.agent_designed["Mass"]          # Units: Kg
            force = self.agent_designed["Force"]        # Units: N
            mu = self.agent_designed["mu"]              # Friction coefficient
            agility = self.agent_designed["Agility"]    # Units: rad/s
        
        # Apply Steering
        self.agent_orientation += steering * agility * self.dt  

        # Apply Acceleration
        acceleration = (throttle * force) / mass # Units: m/s^2

        # Update velocity (x,y)
        self.agent_velocity[0] += math.cos(self.agent_orientation) * acceleration * self.dt
        self.agent_velocity[1] += math.sin(self.agent_orientation) * acceleration * self.dt


    def update_enemy_physics(self):
        # Select strategy
        fx, fy = self.strategy.execute_strategy(self.enemy_position, self.agent_position)
        
        # F = m * a
        accel_x = fx / self.enemy_mass
        accel_y = fy / self.enemy_mass

        self.enemy_velocity[0] += accel_x * self.dt
        self.enemy_velocity[1] += accel_y * self.dt
    

    def handle_collisions(self):
        dx = self.agent_position[0] - self.enemy_position[0]
        dy = self.agent_position[1] - self.enemy_position[1]
        dist_bots = math.sqrt(dx**2 + dy**2)
        min_dist = 1.0 # Radius robot A (0.5) + Radius robot B (0.5)
        
        if dist_bots < min_dist:
            # Collision detected
            # Normal vector (collision direction)
            nx = dx / dist_bots
            ny = dy / dist_bots

            # Relative velocity 
            vx_rel = self.agent_velocity[0] - self.enemy_velocity[0]
            vy_rel = self.agent_velocity[1] - self.enemy_velocity[1]
            vel_along_normal = (vx_rel * nx) + (vy_rel * ny)

            # If its aproaching, calculate the elastic collision response
            if vel_along_normal < 0:
                restitution = 1.1 # Restitution coefficient (Rebot) (>1 -> makes it more "bouncy") 
               
                agent_mass = 5.0
                if self.agent_designed is not None:
                    agent_mass = self.agent_designed["Mass"]  # Units: Kg
                enemy_mass = self.enemy_mass
                
                j = -(1 + restitution) * vel_along_normal
                j /= (1/agent_mass + 1/enemy_mass)
                
                # Apply impulse to both robots
                impulse_x = j * nx
                impulse_y = j * ny
                
                # Update velocities based on impulse
                self.agent_velocity[0] += impulse_x / agent_mass
                self.agent_velocity[1] += impulse_y / agent_mass
                self.enemy_velocity[0] -= impulse_x / enemy_mass
                self.enemy_velocity[1] -= impulse_y / enemy_mass

    
    def apply_movement_and_friction(self):

        if self.agent_designed is None:
            mu = 0.5
        else:
            mu = self.agent_designed["mu"]

        # Friction factor (decrease velocity over time)
        friction_factor = 1.0 - (mu * self.dt) 
        self.agent_velocity *= friction_factor
        self.enemy_velocity *= friction_factor

        # Update positions
        self.agent_position += self.agent_velocity * self.dt
        self.enemy_position += self.enemy_velocity * self.dt

    
    def compute_rewards_and_termination(self):
        dist_agent_center = np.linalg.norm(self.agent_position)
        dist_enemy_center = np.linalg.norm(self.enemy_position)

        reward = 0.0

        # Punishment for time spent
        reward -= 0.1
        
        # Reward for keeping the enemy far from the center
        reward += dist_enemy_center * 0.1
        
        # Warning for being close to the edge
        if dist_agent_center > (self.ring_radius * 0.8):
            reward -= 0.1    
        
        terminated = False  # Episode ended (Victory / Defeat)
        truncated = False   # Episode truncated (Max steps reached)
        
        # --- CHECK DEFEAT CONDITION ---
        if dist_agent_center > self.ring_radius:
            terminated = True
            reward = -100.0
            print(f"DEFEAT! Strategy: {type(self.strategy).__name__}")
        
        # --- CHECK VICTORY CONDITION ---
        elif dist_enemy_center > self.ring_radius:
            terminated = True
            reward = 100.0
            print(f"VICTORY! vs: {type(self.strategy).__name__}")
        
        # --- CHECK TIME LIMIT CONDITION
        if self.current_step_count >= 1000:
            print("SURVIVED!")
            truncated = True
        
        return reward, terminated, truncated