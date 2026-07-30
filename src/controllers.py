"""
controllers.py

Simple PID controller for a single kinematic robot, intentionally built
with the SAME interface as MPCController.compute_control() in the
project's mpc_controller.py:

    control, info = controller.compute_control(current_state, target_position)

where current_state = [x, y, theta, v, omega] and target_position = [x, y].

This means the swap to MPC later is:
    controller = MPCController(...)   # instead of PIDController(...)
and nothing else in simulate.py / formation.py / obstacle_avoidance.py
needs to change.
"""

import math
import numpy as np
from typing import Dict, Tuple


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class PIDController:
    """
    Two coupled PID loops:
      - heading PID -> omega_cmd
      - distance PID -> v_cmd (scaled down while the robot is not yet
        facing the target, so it doesn't drive sideways into obstacles)
    """

    def __init__(self,
                 kp_v: float = 1.2, ki_v: float = 0.0, kd_v: float = 0.05,
                 kp_omega: float = 2.5, ki_omega: float = 0.0, kd_omega: float = 0.1,
                 max_v: float = 0.6, max_omega: float = 1.8,
                 dt: float = 0.05):
        self.kp_v, self.ki_v, self.kd_v = kp_v, ki_v, kd_v
        self.kp_omega, self.ki_omega, self.kd_omega = kp_omega, ki_omega, kd_omega
        self.max_v = max_v
        self.max_omega = max_omega
        self.dt = dt

        self._int_heading = 0.0
        self._prev_heading_err = 0.0
        self._int_dist = 0.0
        self._prev_dist = 0.0

    def reset(self):
        self._int_heading = 0.0
        self._prev_heading_err = 0.0
        self._int_dist = 0.0
        self._prev_dist = 0.0

    def compute_control(self,
                         current_state: np.ndarray,
                         target_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        x, y, theta = current_state[0], current_state[1], current_state[2]
        tx, ty = target_position[0], target_position[1]

        dx, dy = tx - x, ty - y
        distance = math.hypot(dx, dy)
        desired_heading = math.atan2(dy, dx)
        heading_err = normalize_angle(desired_heading - theta)

        # heading PID -> omega
        self._int_heading += heading_err * self.dt
        d_heading = (heading_err - self._prev_heading_err) / self.dt
        omega_cmd = (self.kp_omega * heading_err
                     + self.ki_omega * self._int_heading
                     + self.kd_omega * d_heading)
        omega_cmd = float(np.clip(omega_cmd, -self.max_omega, self.max_omega))
        self._prev_heading_err = heading_err

        # distance PID -> v, damped when not facing the target
        self._int_dist += distance * self.dt
        d_dist = (distance - self._prev_dist) / self.dt
        v_cmd = (self.kp_v * distance
                 + self.ki_v * self._int_dist
                 + self.kd_v * d_dist)
        v_cmd *= max(0.0, math.cos(heading_err))
        v_cmd = float(np.clip(v_cmd, 0.0, self.max_v))
        self._prev_dist = distance

        info = {
            "distance_error": distance,
            "heading_error": heading_err,
            "success": True,
        }
        return np.array([v_cmd, omega_cmd]), info
