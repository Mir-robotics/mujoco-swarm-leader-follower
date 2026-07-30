"""
mpc_controller.py

Short-horizon nonlinear MPC for the same unicycle kinematics used by
PIDController, built with the identical drop-in interface:

    control, info = controller.compute_control(current_state, target_position)

where current_state = [x, y, theta, v, omega] and target_position = [x, y].

Unlike PIDController, this controller *can* also take obstacles directly
into its cost function via set_obstacles() -- this is the "proper" avoidance
path that obstacle_avoidance.py's docstring alludes to (collision_weight /
min_robot_distance below are the fields it mentions). Passing obstacles is
optional: with none set, MPC behaves purely as a better tracker of whatever
target blend_target() hands it, exactly like PID does.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _rollout(state0, u_seq, dt):
    """Roll out unicycle kinematics for a control sequence. u_seq shape (N, 2)."""
    states = np.zeros((len(u_seq) + 1, 3))
    states[0] = state0
    for k in range(len(u_seq)):
        x, y, theta = states[k]
        v, omega = u_seq[k]
        states[k + 1] = [x + v * math.cos(theta) * dt,
                          y + v * math.sin(theta) * dt,
                          theta + omega * dt]
    return states


class MPCController:
    """
    Direct multiple-shooting-free (single shooting) nonlinear MPC solved each
    step with SLSQP over a short horizon. Warm-started from the previous
    solution shifted by one step for speed and smoothness.
    """

    def __init__(self,
                 dt: float = 0.05,
                 horizon: int = 8,
                 max_v: float = 0.6,
                 max_omega: float = 1.8,
                 goal_weight: float = 6.0,
                 control_weight: float = 0.05,
                 smooth_weight: float = 0.15,
                 collision_weight: float = 8.0,
                 min_robot_distance: float = 0.35):
        self.dt = dt
        self.horizon = horizon
        self.max_v = max_v
        self.max_omega = max_omega
        self.goal_weight = goal_weight
        self.control_weight = control_weight
        self.smooth_weight = smooth_weight
        # obstacle-avoidance cost terms -- unused until set_obstacles() is called,
        # at which point MPC bakes avoidance directly into the optimization instead
        # of relying only on obstacle_avoidance.py reshaping the target point.
        self.collision_weight = collision_weight
        self.min_robot_distance = min_robot_distance

        self.obstacles: List = []
        self.robot_radius: float = 0.22

        self._prev_solution: Optional[np.ndarray] = None

    def reset(self):
        self._prev_solution = None

    def set_obstacles(self, obstacles: List, robot_radius: float):
        """Optionally give the optimizer direct knowledge of nearby obstacles."""
        self.obstacles = obstacles
        self.robot_radius = robot_radius

    def _cost(self, u_flat, state0, target):
        u_seq = u_flat.reshape(self.horizon, 2)
        states = _rollout(state0, u_seq, self.dt)

        cost = 0.0
        for k in range(1, self.horizon + 1):
            dx = states[k, 0] - target[0]
            dy = states[k, 1] - target[1]
            weight = self.goal_weight * (1.0 + 0.3 * k / self.horizon)  # emphasize terminal steps
            cost += weight * (dx * dx + dy * dy)

            if self.obstacles:
                px, py = states[k, 0], states[k, 1]
                for obs in self.obstacles:
                    d = math.hypot(px - obs.x, py - obs.y)
                    clearance = d - obs.bounding_radius - self.robot_radius
                    if clearance < self.min_robot_distance:
                        violation = self.min_robot_distance - clearance
                        cost += self.collision_weight * violation * violation

        cost += self.control_weight * np.sum(u_seq[:, 0] ** 2 + u_seq[:, 1] ** 2)
        d_u = np.diff(u_seq, axis=0)
        cost += self.smooth_weight * np.sum(d_u ** 2)
        return cost

    def compute_control(self,
                         current_state: np.ndarray,
                         target_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        state0 = np.array([current_state[0], current_state[1], current_state[2]])
        target = np.asarray(target_position)

        if self._prev_solution is not None:
            warm = np.vstack([self._prev_solution[1:], self._prev_solution[-1:]])
        else:
            warm = np.zeros((self.horizon, 2))
            warm[:, 0] = 0.2
        u0 = warm.flatten()

        bounds = [(0.0, self.max_v), (-self.max_omega, self.max_omega)] * self.horizon

        result = minimize(self._cost, u0, args=(state0, target),
                           method="SLSQP", bounds=bounds,
                           options={"maxiter": 25, "ftol": 1e-3})

        u_seq = result.x.reshape(self.horizon, 2)
        self._prev_solution = u_seq

        v_cmd, omega_cmd = u_seq[0]
        v_cmd = float(np.clip(v_cmd, 0.0, self.max_v))
        omega_cmd = float(np.clip(omega_cmd, -self.max_omega, self.max_omega))

        info = {
            "distance_error": math.hypot(state0[0] - target[0], state0[1] - target[1]),
            "heading_error": _wrap(math.atan2(target[1] - state0[1], target[0] - state0[0]) - state0[2]),
            "success": bool(result.success),
            "solver_cost": float(result.fun),
        }
        return np.array([v_cmd, omega_cmd]), info
