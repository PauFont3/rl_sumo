
def convert_given_points_to_robot_parameters(design_points):
    """
    Recives a list of design points and converts them to robot parameters.
    Order:
    [0]: Mass (Kg)
    [1]: Acceleration (N)
    [2]: Velocity (m/s)
    [3]: Grip (Kg)
    [4]: Agility (Kg)
    """

    mass_pts = design_points[0] + 1
    force_pts = design_points[1] + 1
    # vel_pts  = design_points[2] + 1  # (Reserved for later)
    grip_pts = design_points[3] + 1
    agility_pts   = design_points[4] + 1

    # MASS
    base_mass = 1.0 # Kg
    mass_multiplier = 1.0 # Each point adds 1.0 Kg
    mass = base_mass + (mass_pts * mass_multiplier)

    # FORCE
    base_force = 60.0 # Newtons
    force_multiplier = 10.0 # Each point adds 10.0 N
    max_force = base_force + (force_pts * force_multiplier)

    # GRIP
    base_grip = 0.2 # friction coefficient
    grip_multiplier = 0.16 # Each point adds 0.16 --> (1.0 - 0.2) / 5 levels = 0.16
    mu = base_grip + (grip_pts * grip_multiplier)

    # AGILITY
    base_steering = 1.0 # rad/s
    steering_multiplier = 0.5 # Each point adds 0.5 rad/s
    agility = base_steering + (agility_pts * steering_multiplier)

    return {
        "Mass": mass,
        "Force": max_force,
        "mu": mu,
        "Agility": agility
    }