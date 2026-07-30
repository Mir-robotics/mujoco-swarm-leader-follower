"""
simulate.py

Five mobile robots in a queue (leader + four followers), crossing a field
scattered with ~15-20 obstacles of mixed shapes (box / cylinder / sphere),
using a simple PID controller + potential-field obstacle avoidance.

Run:
    python3 simulate.py                 # opens the MuJoCo interactive viewer
    python3 simulate.py --headless      # no viewer, just logs + saves a plot
    python3 simulate.py --seed 7 --num-obstacles 20

Upgrading to MPC later:
    Swap `PIDController(...)` for the project's `MPCController(...)` --
    the compute_control(state, target) interface is identical, see
    controllers.py's module docstring.
"""

import argparse
import os
import csv
import math
import time
import numpy as np

# Import mujoco
try:
    import mujoco
    print(f"[OK] mujoco version: {mujoco.__version__}")
except ImportError as e:
    print(f"[ERROR] Failed to import mujoco: {e}")
    raise

from model_builder import WorldConfig, build_world
from controllers import PIDController, normalize_angle
from obstacle_avoidance import blend_target, min_clearance
from formation import queue_target

from perception import CameraPerception, PerceivedObstacle

# Colors for 5 robots
ROBOT_COLORS = [
    "0.85 0.15 0.15 1",  # leader (red)
    "0.15 0.35 0.85 1",  # follower_1 (blue)
    "0.15 0.65 0.25 1",  # follower_2 (green)
    "0.85 0.85 0.15 1",  # follower_3 (yellow)
    "0.85 0.15 0.85 1",  # follower_4 (magenta)
]
ROBOT_NAMES = ["leader", "follower_1", "follower_2", "follower_3", "follower_4"]


def yaw_to_quat(yaw: float):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def set_robot_pose(data, model, body_name, x, y, z, yaw):
    qpos_adr = model.jnt_qposadr[model.joint(f"{body_name}_free").id]
    data.qpos[qpos_adr:qpos_adr + 3] = [x, y, z]
    data.qpos[qpos_adr + 3:qpos_adr + 7] = yaw_to_quat(yaw)


def run(args):
    cfg = WorldConfig(
        field_length=args.field_length,
        field_width=args.field_width,
        num_obstacles=args.num_obstacles,
        seed=args.seed,
        robot_radius=0.22,
    )

    gap = 0.55  # desired spacing between robots in the queue
    # NOTE: must stay comfortably above 2 * cfg.robot_radius (0.44 m here),
    # otherwise consecutive robots overlap even at rest, and get stuck on
    # each other while turning through the chokepoint gates.
    start_y = 0.0
    # Dynamically generate starting positions for all robots
    robot_starts = []
    for i in range(len(ROBOT_NAMES)):
        x = -i * gap
        robot_starts.append((x, start_y, ROBOT_COLORS[i]))

    xml, obstacles = build_world(cfg, robot_starts)
    xml_path = os.path.join(os.path.dirname(__file__), "generated_world.xml")
    
    # ===== اصلاح: نوشتن فایل با encoding=utf-8 =====
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)

    # Use the imported mujoco module
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    dt = args.dt
    model.opt.timestep = dt

    # Per-robot kinematic state [x, y, theta, v, omega]
    states = []
    for (x, y, _rgba) in robot_starts:
        states.append(np.array([x, y, 0.0, 0.0, 0.0]))

    controllers = [PIDController(dt=dt) for _ in ROBOT_NAMES]

    goal = np.array([cfg.field_length - 1.0, 0.0])

    # logging
    log_path = os.path.join(os.path.dirname(__file__), "trajectory_log.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["t", "robot", "x", "y", "theta", "v", "omega", "min_clearance"])

    n_steps = args.duration_steps
    collision_events = 0
    min_clearance_seen = math.inf
    arrived = [False] * len(ROBOT_NAMES)

    viewer_ctx = None
    if not args.headless:
        try:
            # Import viewer separately to avoid namespace conflicts
            from mujoco import viewer as mujoco_viewer
            viewer_ctx = mujoco_viewer.launch_passive(model, data)

            # Top-down (bird's-eye) camera, framing the whole field
            viewer_ctx.cam.lookat[:] = [cfg.field_length / 2.0, 0.0, 0.0]
            viewer_ctx.cam.distance = max(cfg.field_length, cfg.field_width) * 0.85
            viewer_ctx.cam.elevation = -90
            viewer_ctx.cam.azimuth = 90

            print("[OK] Interactive viewer launched (top-down view)")
            print("[info] showing the full field for a couple seconds before the robots start moving...")

            # Let the user see the full, static scene (obstacles + robots) from
            # above before anything starts moving.
            preview_seconds = 2.5
            t_preview_start = time.perf_counter()
            while viewer_ctx.is_running() and (time.perf_counter() - t_preview_start) < preview_seconds:
                mujoco.mj_forward(model, data)
                viewer_ctx.sync()
                time.sleep(0.02)
        except Exception as e:
            print(f"[warn] could not open interactive viewer ({e}); running headless.")
            viewer_ctx = None

    t = 0.0
    step = 0
    wall_clock_start = time.perf_counter()
    try:
        for step in range(n_steps):
            # --- compute nominal target for each robot ---
            nominal_targets = [goal]
            for i in range(1, len(states)):
                nominal_targets.append(queue_target(states[i - 1], gap))

            # --- avoidance + control + kinematic integration ---
            for i, name in enumerate(ROBOT_NAMES):
                pos = states[i][:2]
                safe_target = blend_target(pos, nominal_targets[i], obstacles,
                                            robot_radius=cfg.robot_radius)
                control, _info = controllers[i].compute_control(states[i], safe_target)
                v_cmd, omega_cmd = control

                theta = states[i][2] + omega_cmd * dt
                x = states[i][0] + v_cmd * math.cos(states[i][2]) * dt
                y = states[i][1] + v_cmd * math.sin(states[i][2]) * dt
                states[i] = np.array([x, y, normalize_angle(theta), v_cmd, omega_cmd])

                clearance = min_clearance(states[i][:2], obstacles, cfg.robot_radius)
                min_clearance_seen = min(min_clearance_seen, clearance)
                if clearance < 0:
                    collision_events += 1

                set_robot_pose(data, model, f"robot{i}", x, y, cfg.robot_radius, states[i][2])
                writer.writerow([f"{t:.2f}", name, f"{x:.3f}", f"{y:.3f}",
                                  f"{states[i][2]:.3f}", f"{v_cmd:.3f}", f"{omega_cmd:.3f}",
                                  f"{clearance:.3f}"])

            mujoco.mj_forward(model, data)

            if viewer_ctx is not None:
                if not viewer_ctx.is_running():
                    break
                viewer_ctx.sync()
                # real-time pacing: without this the loop runs as fast as the
                # CPU allows and the whole crossing flashes by in ~1-2 seconds
                target_wall_time = wall_clock_start + t + dt
                now = time.perf_counter()
                if target_wall_time > now:
                    time.sleep(target_wall_time - now)

            t += dt

            # A robot is "arrived" once it's within a radius of the goal.
            # Robots further back in the queue end up parked in a line
            # BEHIND the leader rather than exactly on the goal point, so
            # their arrival radius has to be a bit more generous or they'd
            # never register as "arrived" and the sim would just run to
            # duration_steps every time.
            for i in range(len(states)):
                if not arrived[i]:
                    threshold = 0.3 + i * gap * 1.5
                    if np.linalg.norm(states[i][:2] - goal) < threshold:
                        arrived[i] = True
                        print(f"[info] {ROBOT_NAMES[i]} reached the goal region "
                              f"at t={t:.1f}s (step {step})")

            # stop only once EVERY robot (not just the leader) has arrived
            if all(arrived):
                print(f"[info] all robots reached the goal at t={t:.1f}s (step {step})")
                break
    except KeyboardInterrupt:
        print("\n[info] Simulation interrupted by user")
    finally:
        log_file.close()
        if viewer_ctx is not None and viewer_ctx.is_running():
            print("[info] run finished -- the top-down view stays open; close the viewer window yourself when done")
            try:
                while viewer_ctx.is_running():
                    mujoco.mj_forward(model, data)
                    viewer_ctx.sync()
                    time.sleep(0.02)
            except KeyboardInterrupt:
                pass
        if viewer_ctx is not None:
            viewer_ctx.close()
            
            
    # Initialize perception for each robot
    perceptions = []
    if not args.headless:
        try:
            for i in range(len(ROBOT_NAMES)):
                perceptions.append(CameraPerception(model))
            print(f"[OK] Perception initialized for {len(ROBOT_NAMES)} robots")
        except Exception as e:
            print(f"[WARN] Perception initialization failed: {e}")
            perceptions = []        

    print(f"[info] simulated {step + 1} steps ({t:.1f}s)")
    if all(arrived):
        print("[info] every robot reached the goal.")
    else:
        missing = [ROBOT_NAMES[i] for i, a in enumerate(arrived) if not a]
        print(f"[warn] duration_steps ran out before these robots arrived: {missing}")
        print("[warn] try --duration-steps with a larger value, or check for a stuck robot "
              "(see min_clearance_seen / trajectory_plot.png).")
    print(f"[info] collision events (clearance<0 samples): {collision_events}")
    print(f"[info] minimum clearance ever seen: {min_clearance_seen:.3f} m")
    print(f"[info] trajectory log saved to {log_path}")

    return log_path, obstacles, cfg


def plot_result(log_path, obstacles, cfg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as e:
        print(f"[ERROR] matplotlib or pandas not installed: {e}")
        print("[ERROR] Please run: pip install matplotlib pandas")
        return None

    df = pd.read_csv(log_path)
    fig, ax = plt.subplots(figsize=(10, 5))

    for obs in obstacles:
        if obs.shape == "sphere":
            circ = plt.Circle((obs.x, obs.y), obs.bounding_radius, color="#3fa34d", alpha=0.5)
            ax.add_patch(circ)
        elif obs.shape == "cylinder":
            circ = plt.Circle((obs.x, obs.y), obs.bounding_radius, color="#3372cc", alpha=0.5)
            ax.add_patch(circ)
        else:
            hx, hy, _ = obs.size
            rect = plt.Rectangle((obs.x - hx, obs.y - hy), 2 * hx, 2 * hy,
                                  color="#c04e33", alpha=0.5)
            ax.add_patch(rect)

    # Colors for plotting (matching the robot colors)
    colors = {
        "leader": "#d92626",
        "follower_1": "#2657d9",
        "follower_2": "#26a626",
        "follower_3": "#d9d926",
        "follower_4": "#d926d9",
    }
    for name, group in df.groupby("robot"):
        ax.plot(group["x"], group["y"], label=name, color=colors.get(name), linewidth=2)
        ax.plot(group["x"].iloc[0], group["y"].iloc[0], "o", color=colors.get(name))

    ax.set_xlim(-1, cfg.field_length + 1)
    ax.set_ylim(-cfg.field_width / 2 - 1, cfg.field_width / 2 + 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.set_title("Leader-follower queue crossing a randomized obstacle field")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    #out_path = os.path.join(os.path.dirname(log_path), "trajectory_plot.png")
    #fig.savefig(out_path, dpi=150, bbox_inches="tight")
    #print(f"[info] trajectory plot saved to {out_path}")
    #plt.close(fig)
    #return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                         help="skip the interactive viewer (useful with no display)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-obstacles", type=int, default=12)
    parser.add_argument("--field-length", type=float, default=20.0)
    parser.add_argument("--field-width", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--duration-steps", type=int, default=2000)
    parser.add_argument("--plot", action="store_true", default=True)
    args = parser.parse_args()

    log_path, obstacles, cfg = run(args)
    if args.plot and log_path:
        plot_result(log_path, obstacles, cfg)