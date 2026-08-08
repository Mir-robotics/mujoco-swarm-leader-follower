# MuJoCo Multi-Robot Leader-Follower Swarm

A simulated swarm of mobile robots that crosses a randomized, obstacle-scattered
field in a leader-follower queue, using potential-field obstacle avoidance,
onboard-camera perception, and swappable PID / MPC controllers — built on
[MuJoCo](https://mujoco.org/).

![status](https://img.shields.io/badge/status-active--development-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## Demo

▶️ Demo in YouTube: **[https://youtube.com/shorts/gyOKlvq-LHQ?feature=share]**

## Overview

Five robots (one leader, four followers) drive in a queue from one end of a
field to the other. The field is scattered with 12+ randomly placed box /
cylinder / sphere obstacles, plus optional serpentine "chokepoint" gates that
force the queue to zigzag across the whole course. Each robot avoids
collisions with potential-field target-reshaping, is driven by a PID (and,
optionally, MPC) controller, and can perceive obstacles through its own
onboard camera instead of relying on ground-truth positions.

## Features

- **Leader-follower queue formation** — each follower tracks a point directly
  behind the robot ahead of it, offset along that robot's current heading.
- **Randomized obstacle field** — configurable count, field size, and random
  seed; box / cylinder / sphere obstacles with collision-safe placement.
- **Serpentine chokepoint gates** — near-solid walls spanning the field with a
  single narrow, alternating-side opening, forcing genuine navigation instead
  of open-field driving.
- **Potential-field obstacle avoidance** — reshapes each robot's target point
  away from nearby obstacles; controller-agnostic by design (works identically
  whichever controller is plugged in).
- **Two interchangeable control strategies**, both behind the exact same
  `compute_control(state, target)` interface:
  - `PIDController` — coupled heading/distance PID loops (the default).
  - `MPCController` — short-horizon nonlinear MPC (SLSQP, warm-started) that
    can additionally bake obstacle avoidance directly into its cost function
    via `set_obstacles()`, rather than only reacting to a reshaped target.
- **Onboard camera perception** (`perception.py`) — renders RGB + depth from
  each robot's forward-facing MuJoCo camera, color-segments obstacle pixels,
  and back-projects detections into world-frame position/radius estimates,
  as a real (occlusion- and range-limited) alternative to ground-truth
  obstacle knowledge.
- **Interactive top-down viewer** or **headless batch mode**, with CSV
  trajectory logging and an auto-generated matplotlib plot of the run.

## Repository Structure

```
mujoco-swarm-leader-follower/
├── src/
│   ├── simulate.py            # entry point: builds the world, runs the sim loop, logs + plots
│   ├── model_builder.py       # MuJoCo XML generation: field, obstacles, chokepoint gates, robots + cameras
│   ├── formation.py           # leader-follower queue target computation
│   ├── obstacle_avoidance.py  # potential-field target reshaping + clearance checks
│   ├── controllers.py         # PIDController (default control strategy)
│   ├── mpc_controller.py      # MPCController (alternative control strategy, same interface)
│   └── perception.py          # onboard-camera obstacle detection (CameraPerception)
├── results/                   # trajectory_log.csv / trajectory_plot.png land here when you point outputs here
├── docs/
│   └── media/                 # screenshots, GIFs, demo video assets for the README
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/mujoco-swarm-leader-follower.git
cd mujoco-swarm-leader-follower
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Requires Python 3.10+. MuJoCo's offscreen renderer (used by `perception.py`)
needs a working OpenGL backend — on a headless machine, set:

```bash
export MUJOCO_GL=egl     # or osmesa if EGL isn't available
```

## Usage

```bash
# interactive MuJoCo viewer (top-down)
python3 src/simulate.py

# headless run: logs + plot only, no display needed
python3 src/simulate.py --headless

# customize the field
python3 src/simulate.py --headless --seed 7 --num-obstacles 20 \
    --field-length 25 --field-width 10 --duration-steps 3000
```

| Flag | Default | Description |
|---|---|---|
| `--headless` | off | Skip the interactive viewer; just simulate, log, and plot |
| `--seed` | `42` | RNG seed for obstacle placement |
| `--num-obstacles` | `12` | Number of scattered (non-gate) obstacles |
| `--field-length` | `20.0` | Field length in meters |
| `--field-width` | `8.0` | Field width in meters |
| `--dt` | `0.05` | Simulation timestep (s) |
| `--duration-steps` | `2000` | Max steps before the run is cut off |
| `--plot` | on | Save a trajectory plot after the run |

Outputs (`generated_world.xml`, `trajectory_log.csv`, `trajectory_plot.png`)
are written next to `simulate.py` by default.

## How It Works

**Formation.** `queue_target()` gives each follower a nominal target point a
fixed `gap` behind the robot ahead of it, along that robot's current heading —
so the queue naturally follows the leader's path rather than beelining
independently.

**Avoidance.** `blend_target()` combines an attractive vector toward the
nominal target with a repulsive vector away from every obstacle within an
influence radius, and hands the *reshaped* point to the controller. This
keeps avoidance logic completely decoupled from which controller is driving —
PID and MPC both consume the same `compute_control(state, safe_target)` call.

**Control.** `PIDController` runs two coupled PID loops (heading → ω, distance
→ v). `MPCController` instead rolls out unicycle kinematics over a short
horizon and solves for the control sequence that minimizes tracking + control
effort + smoothness cost (SLSQP, warm-started from the previous step). It also
exposes `set_obstacles()` so obstacle clearance can be penalized directly
inside the optimization, rather than only through a reshaped target.

**Perception.** `CameraPerception.detect()` renders each robot's onboard
camera, segments obstacle-colored pixels (a palette deliberately kept distinct
from the robot colors), connected-component-labels the mask, reads depth at
each blob's centroid, and back-projects through the pinhole camera model
(using the camera's `cam_xpos`/`cam_xmat` extrinsics) into a world-frame
position + radius estimate — pushed back along the camera ray to approximate
the obstacle's center rather than its near surface.

## Known Limitations / Roadmap

- `MPCController` and `CameraPerception` are fully implemented but not yet
  wired into `simulate.py`'s main loop by a CLI switch — today the sim always
  drives with `PIDController` against the ground-truth obstacle list. Next
  step: add a `--controller {pid,mpc}` flag and a `--vision` flag that swaps
  the ground-truth obstacle list for each robot's own `CameraPerception`
  detections during avoidance.
- Vision-based avoidance means obstacles outside camera FOV/range are
  invisible until they come into view — this is intentional (realistic
  perception) but changes robot behavior versus the ground-truth mode, and
  hasn't been benchmarked against it yet.
- No dynamic re-formation (e.g., switching queue → line-abreast) yet — only
  the queue formation is implemented.

## License

MIT — see [LICENSE](LICENSE).
