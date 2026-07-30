"""
obstacle_avoidance.py

Simple artificial-potential-field avoidance.

Design choice: instead of touching the controller (PID today, MPC later),
avoidance works by *reshaping the target point* that gets fed into
compute_control(). That keeps controllers.py generic and means the exact
same blend_target() call will still work once the project's MPCController
is dropped in -- MPC just receives an already-safer target_position.

(Note for later: once you upgrade to MPCController, the "proper" way is to
bake collision avoidance directly into its cost function -- the
collision_weight / min_robot_distance fields are already declared in that
class but never used in _setup_optimization_problem(). This module is the
quick path to get something working first.)
"""

import math
import numpy as np
from typing import List
from model_builder import Obstacle


def repulsive_vector(pos: np.ndarray,
                      obstacles: List[Obstacle],
                      influence_radius: float,
                      gain: float,
                      robot_radius: float) -> np.ndarray:
    """Sum of repulsive vectors from all obstacles within influence_radius."""
    rep = np.zeros(2)
    for obs in obstacles:
        d = pos - np.array([obs.x, obs.y])
        dist = float(np.linalg.norm(d)) - obs.bounding_radius - robot_radius
        dist = max(dist, 0.05)  # avoid singularity
        if dist < influence_radius:
            strength = gain * (1.0 / dist - 1.0 / influence_radius) / (dist ** 2)
            rep += strength * (d / (np.linalg.norm(d) + 1e-6))
    return rep


def blend_target(pos: np.ndarray,
                  nominal_target: np.ndarray,
                  obstacles: List[Obstacle],
                  robot_radius: float,
                  influence_radius: float = 1.3,
                  repulsion_gain: float = 0.35,
                  lookahead_cap: float = 2.5) -> np.ndarray:
    """
    Returns a new target point that blends the attraction towards
    nominal_target with repulsion away from nearby obstacles.
    """
    to_goal = nominal_target - pos
    goal_dist = float(np.linalg.norm(to_goal))
    attractive_dir = to_goal / goal_dist if goal_dist > 1e-6 else np.zeros(2)

    rep = repulsive_vector(pos, obstacles, influence_radius, repulsion_gain, robot_radius)

    combined = attractive_dir + rep
    norm = float(np.linalg.norm(combined))
    if norm < 1e-6:
        return nominal_target  # nothing to avoid, no strong pull either

    combined_dir = combined / norm
    lookahead = min(goal_dist, lookahead_cap)
    return pos + combined_dir * lookahead


def min_clearance(pos: np.ndarray, obstacles: List[Obstacle], robot_radius: float) -> float:
    """Smallest gap between the robot's hull and any obstacle's hull (can be negative = collision)."""
    best = math.inf
    for obs in obstacles:
        d = float(np.linalg.norm(pos - np.array([obs.x, obs.y])))
        clearance = d - obs.bounding_radius - robot_radius
        best = min(best, clearance)
    return best
