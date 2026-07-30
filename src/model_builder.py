"""
model_builder.py

Builds:
  - the MuJoCo world XML (ground plane, boundary walls, a randomized
    obstacle field, three kinematic robot bodies with a free joint each,
    and a forward-facing onboard camera per robot for the vision module), and
  - the ground-truth `Obstacle` list used for evaluation (collision / clearance
    logging) and, optionally, as a fallback when vision is disabled.

Obstacle colors are chosen to NOT overlap with the robot colors
(red / blue / green in simulate.py), so perception.py can tell "obstacle"
from "another robot" using color alone.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple


# shape -> rgba, همه موانع به رنگ مشکی
SHAPE_RGBA = {
    "box":      "1.00 0.55 0.10 1",  # vibrant orange
    "cylinder": "1.00 0.55 0.10 1",
    "sphere":   "1.00 0.55 0.10 1",
}


@dataclass
class Obstacle:
    shape: str                 # "box" | "cylinder" | "sphere"
    x: float
    y: float
    z: float
    size: Tuple[float, float, float]   # box: half-extents; cylinder: (radius, half-height, 0); sphere: (radius, 0, 0)
    bounding_radius: float      # 2D (x,y) bounding radius used by avoidance / clearance checks
    rgba: str = ""

    def __post_init__(self):
        if not self.rgba:
            self.rgba = SHAPE_RGBA[self.shape]


@dataclass
class WorldConfig:
    field_length: float = 20.0
    field_width: float = 8.0
    num_obstacles: int = 16  # scattered obstacles; the chokepoint gates add ~25-35 more on top of this
    seed: int = 42
    robot_radius: float = 0.22
    wall_height: float = 0.4
    min_obstacle_radius: float = 0.25
    max_obstacle_radius: float = 0.65
    # keep a clear corridor around x < corridor_clear_x so the robots don't spawn inside an obstacle
    corridor_clear_x: float = 2.0
    # keep a clear zone around the goal so the run doesn't stall right at the finish
    goal_clear_radius: float = 1.2

    # --- narrow-passage chokepoints (a serpentine of gates, not one doorway) ---
    # Several near-solid walls spanning almost the full field, each sealed
    # right up to the side rails (no sneaking around the ends) and each with
    # a single narrow gap. Gaps alternate side-to-side so the robots have to
    # zigzag the whole length of the course instead of crossing one gap and
    # then having open space again.
    add_chokepoint: bool = True
    num_chokepoints: int = 2                    # how many gates along the course
    chokepoint_span: Tuple[float, float] = (0.3, 0.85)  # gates spread across this fraction of field_length
    chokepoint_gap_width: float = 2.0           # width of each passable opening (m)
    chokepoint_alt_offset: float = 1.0          # how far alternating gates sit from centerline (the "zigzag")
    chokepoint_obstacle_radius: float = 0.32    # size of the wall-forming obstacles
    chokepoint_x_jitter: float = 0.2            # small x jitter so each wall isn't perfectly flat
    chokepoint_edge_margin: float = 0.12        # wall pieces get placed this close to the side rails -- small = fully sealed
    chokepoint_gap_between_pieces: float = 0.02 # tiny gap between adjacent wall pieces to prevent overlap (keeps wall solid)

    # scattered (non-wall) obstacles filling the rest of the corridor
    obstacle_margin: float = 0.08  # min gap left between neighboring obstacles when scattering (smaller = denser)


def _random_shape_and_size(rng: random.Random, cfg: WorldConfig):
    shape = rng.choice(["box", "cylinder", "sphere"])
    r = rng.uniform(cfg.min_obstacle_radius, cfg.max_obstacle_radius)
    if shape == "box":
        hx = r * rng.uniform(0.7, 1.0)
        hy = r * rng.uniform(0.7, 1.0)
        hz = rng.uniform(0.2, 0.5)
        size = (hx, hy, hz)
        bounding_radius = math.hypot(hx, hy)
    elif shape == "cylinder":
        radius = r
        half_h = rng.uniform(0.2, 0.5)
        size = (radius, half_h, 0.0)
        bounding_radius = radius
    else:  # sphere
        radius = r
        size = (radius, 0.0, 0.0)
        bounding_radius = radius
    return shape, size, bounding_radius


def _chokepoint_walls(cfg: WorldConfig, rng: random.Random):
    """Build several near-solid walls spanning almost the whole field, each
    sealed to the side rails and each with one narrow gap. Gaps alternate
    left/right of centerline so the course is a zigzag, not a single
    doorway. Uses cylinders/spheres only (not boxes) so every wall piece has
    an exact, predictable bounding_radius -- that keeps each opening close
    to cfg.chokepoint_gap_width instead of drifting with random box
    diagonals.

    Returns (wall_obstacles, gate_zones) where gate_zones is a list of
    (x_center, gap_lo, gap_hi) used elsewhere to keep scattered obstacles
    out of the openings.
    """
    if not cfg.add_chokepoint or cfg.num_chokepoints <= 0:
        return [], []

    half_w = cfg.field_width / 2.0
    x_lo_frac, x_hi_frac = cfg.chokepoint_span
    n = cfg.num_chokepoints
    if n == 1:
        x_centers = [cfg.field_length * (x_lo_frac + x_hi_frac) / 2.0]
    else:
        x_centers = [
            cfg.field_length * (x_lo_frac + (x_hi_frac - x_lo_frac) * i / (n - 1))
            for i in range(n)
        ]

    r = cfg.chokepoint_obstacle_radius
    # فاصله‌ی بین مراکز موانع: دو برابر شعاع + یک فاصله‌ی بسیار کوچک (تا هم‌پوشانی نداشته باشند)
    spacing = 2 * r + cfg.chokepoint_gap_between_pieces
    wall: List[Obstacle] = []
    gate_zones = []

    for gate_idx, x_center in enumerate(x_centers):
        side = 1 if gate_idx % 2 == 0 else -1
        gap_y = side * cfg.chokepoint_alt_offset if n > 1 else 0.0
        gap_lo = gap_y - cfg.chokepoint_gap_width / 2.0
        gap_hi = gap_y + cfg.chokepoint_gap_width / 2.0
        gate_zones.append((x_center, gap_lo, gap_hi))

        # مکان‌های شروع و پایان دیواره: دقیقاً تا لبه‌ی زمین (با احتساب شعاع مانع)
        start_y = -half_w + r
        end_y = half_w - r

        y = start_y
        while y <= end_y:
            # اگر در محدوده‌ی دروازه باشیم، این نقطه را رد می‌کنیم
            if gap_lo <= y <= gap_hi:
                y += spacing
                continue

            x = x_center + rng.uniform(-cfg.chokepoint_x_jitter, cfg.chokepoint_x_jitter)
            # شکل را تصادفی انتخاب می‌کنیم (استوانه یا کره)
            shape = rng.choice(["cylinder", "sphere"])
            if shape == "cylinder":
                half_h = rng.uniform(0.25, 0.45)
                size = (r, half_h, 0.0)
                z = half_h
            else:
                size = (r, 0.0, 0.0)
                z = r

            wall.append(Obstacle(shape=shape, x=x, y=y, z=z, size=size, bounding_radius=r))
            y += spacing

    return wall, gate_zones


def _overlaps(x, y, bounding_radius, others, margin=0.15):
    for o in others:
        d = math.hypot(x - o.x, y - o.y)
        if d < bounding_radius + o.bounding_radius + margin:
            return True
    return False


def _sample_obstacles(cfg: WorldConfig, goal_xy, preplaced: List[Obstacle] = None,
                       gate_zones=None) -> List[Obstacle]:
    rng = random.Random(cfg.seed)
    obstacles: List[Obstacle] = list(preplaced) if preplaced else []
    n_preplaced = len(obstacles)
    attempts = 0
    max_attempts = cfg.num_obstacles * 200

    half_w = cfg.field_width / 2.0
    gate_zones = gate_zones or []
    chokepoint_band = 1.5  # x-band around each gate kept free of extra random clutter

    while len(obstacles) < cfg.num_obstacles + n_preplaced and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(cfg.corridor_clear_x, cfg.field_length - 1.0)
        y = rng.uniform(-half_w + 0.4, half_w - 0.4)

        if math.hypot(x - goal_xy[0], y - goal_xy[1]) < cfg.goal_clear_radius:
            continue

        # keep every gate opening free so random clutter can't re-block it
        if any(abs(x - gx) < chokepoint_band and glo - 0.3 < y < ghi + 0.3
               for gx, glo, ghi in gate_zones):
            continue

        shape, size, bounding_radius = _random_shape_and_size(rng, cfg)
        if _overlaps(x, y, bounding_radius, obstacles, margin=cfg.obstacle_margin):
            continue

        z = size[2] if shape == "box" else (size[1] if shape == "cylinder" else 0.0)
        if shape == "sphere":
            z = size[0]
        elif shape == "cylinder":
            z = size[1]
        else:
            z = size[2]

        obstacles.append(Obstacle(shape=shape, x=x, y=y, z=z, size=size,
                                   bounding_radius=bounding_radius))

    return obstacles


def _obstacle_geom_xml(obs: Obstacle, idx: int) -> str:
    if obs.shape == "box":
        hx, hy, hz = obs.size
        return (f'<body name="obstacle_{idx}" pos="{obs.x} {obs.y} {hz}">'
                f'<geom type="box" size="{hx} {hy} {hz}" rgba="{obs.rgba}"/></body>')
    elif obs.shape == "cylinder":
        radius, half_h, _ = obs.size
        return (f'<body name="obstacle_{idx}" pos="{obs.x} {obs.y} {half_h}">'
                f'<geom type="cylinder" size="{radius} {half_h}" rgba="{obs.rgba}"/></body>')
    else:  # sphere
        radius = obs.size[0]
        return (f'<body name="obstacle_{idx}" pos="{obs.x} {obs.y} {radius}">'
                f'<geom type="sphere" size="{radius}" rgba="{obs.rgba}"/></body>')


def _robot_body_xml(name: str, x: float, y: float, z: float, rgba: str, robot_radius: float,
                     cam_fovy: float = 80.0) -> str:
    # Camera is mounted facing the body's local +X axis (the robot's forward
    # direction in simulate.py's kinematics). xyaxes below rotates the default
    # camera frame (which looks along local -Z) so its forward axis is +X:
    #   local X (right) = body -Y, local Y (up) = body +Z, local Z (backward) = body -X
    return f'''
    <body name="{name}" pos="{x} {y} {z}">
      <joint name="{name}_free" type="free"/>
      <geom type="box" size="{robot_radius} {robot_radius * 0.7} {robot_radius * 0.5}" rgba="{rgba}"/>
      <geom type="box" size="0.05 0.05 0.05" pos="{robot_radius * 0.8} 0 0" rgba="1 1 1 1"/>
      <camera name="{name}_cam" pos="{robot_radius * 0.9} 0 {robot_radius * 0.6}"
              xyaxes="0 -1 0  0 0 1" fovy="{cam_fovy}"/>
    </body>'''


def build_world(cfg: WorldConfig, robot_starts: List[Tuple[float, float, str]]):
    goal_xy = (cfg.field_length - 1.0, 0.0)
    wall, gate_zones = _chokepoint_walls(cfg, random.Random(cfg.seed + 1))
    obstacles = _sample_obstacles(cfg, goal_xy, preplaced=wall, gate_zones=gate_zones)

    half_w = cfg.field_width / 2.0
    wall_t = 0.01

    # ===== زمین بزرگ با texture چهارخانه و skybox (با texrepeat کاهش‌یافته برای درشتی خانه‌ها) =====
    asset = f'''
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="1 1" reflectance="0.2"/>   <!-- کاهش texrepeat برای درشت‌تر شدن خانه‌ها -->
  </asset>
'''

    # زمین با اندازه‌ی بسیار بزرگ (۱۰۰ متر) تا کل صحنه را بپوشاند
    ground = f'''
    <geom name="ground" type="plane" pos="{cfg.field_length / 2} 0 0"
          size="100 100 0.1" material="groundplane"/>
    '''

    # ===== حذف دیوارهای کناری =====
    walls = ""   # خالی کردن متغیر walls

    robot_bodies = []
    for i, (x, y, rgba) in enumerate(robot_starts):
        robot_bodies.append(_robot_body_xml(f"robot{i}", x, y, cfg.robot_radius, rgba, cfg.robot_radius))

    obstacle_bodies = "\n".join(_obstacle_geom_xml(o, i) for i, o in enumerate(obstacles))

    xml = f'''<mujoco model="swarm_field">
  <option timestep="0.05" gravity="0 0 -9.81"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4"/>
  </visual>
  {asset}
  <worldbody>
    <light diffuse="1 1 1" pos="{cfg.field_length / 2} 0 6" dir="0 0 -1" directional="true"/>
    {ground}
    {walls}
    {obstacle_bodies}
    {"".join(robot_bodies)}
  </worldbody>
</mujoco>'''

    return xml, obstacles