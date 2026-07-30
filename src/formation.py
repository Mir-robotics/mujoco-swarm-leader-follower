"""
formation.py

Queue ("پشت سر هم") formation: each follower's nominal target is a point
directly behind the robot ahead of it, offset along that robot's current
heading.
"""

import math
import numpy as np


def queue_target(predecessor_state: np.ndarray, gap: float) -> np.ndarray:
    """
    predecessor_state: [x, y, theta, v, omega]
    Returns the [x, y] point `gap` meters behind the predecessor, along
    its heading.
    """
    x, y, theta = predecessor_state[0], predecessor_state[1], predecessor_state[2]
    tx = x - gap * math.cos(theta)
    ty = y - gap * math.sin(theta)
    return np.array([tx, ty])
