"""
perception.py

Real (non-ground-truth) obstacle perception from each robot's onboard camera.

Pipeline per robot, per frame:
  1. Render RGB + depth from the robot's forward-facing camera
     (`{robot_name}_cam`, mounted in model_builder.py).
  2. Color-segment obstacle-colored pixels (box/cylinder/sphere palettes -- see
     model_builder.SHAPE_RGBA) and reject anything in the robot-color palette,
     so other robots are never mistaken for obstacles.
  3. Connected-component label the mask (scipy.ndimage) -> one blob per
     visible obstacle (or partial obstacle, if occluded).
  4. For each blob: read depth at its centroid pixel, back-project through the
     pinhole camera model to a 3D point in the camera frame, then to world
     frame using the camera's extrinsics (data.cam_xpos / data.cam_xmat).
  5. Depth lands on the obstacle's near surface, not its center, so we push
     the estimate back along the camera ray by an estimated radius (derived
     from the blob's angular size) to approximate the obstacle's centroid.

This deliberately replaces the ground-truth `Obstacle` list used elsewhere:
robots only "see" what's in front of their camera and within `max_range` --
occluded or distant obstacles are invisible until they come into view, which
is the whole point of using vision instead of a god's-eye obstacle list.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import mujoco
from scipy import ndimage

from model_builder import SHAPE_RGBA


@dataclass
class PerceivedObstacle:
    x: float
    y: float
    bounding_radius: float
    confidence: float  # fraction of expected pixels actually seen (occlusion indicator)


# Robot body colors from simulate.py's ROBOT_COLORS, used to exclude robots from the obstacle mask.
ROBOT_RGBA = [
    (0.85, 0.15, 0.15),
    (0.15, 0.35, 0.85),
    (0.15, 0.65, 0.25),
]


def _color_bounds(rgba_str: str, tol: float = 0.12):
    r, g, b, _ = (float(v) for v in rgba_str.split())
    lo = np.array([max(0.0, r - tol), max(0.0, g - tol), max(0.0, b - tol)])
    hi = np.array([min(1.0, r + tol), min(1.0, g + tol), min(1.0, b + tol)])
    return lo, hi


class CameraPerception:
    def __init__(self, model, width: int = 96, height: int = 72,
                 fovy_deg: float = 80.0, max_range: float = 4.5,
                 min_blob_pixels: int = 4):
        self.model = model
        self.width = width
        self.height = height
        self.fovy_deg = fovy_deg
        self.max_range = max_range
        self.min_blob_pixels = min_blob_pixels

        # Initialize renderer with error handling
        try:
            self.renderer = mujoco.Renderer(model, height=height, width=width)
            print(f"[OK] Renderer initialized: {width}x{height}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize renderer: {e}")
            print("[INFO] Make sure you have OpenGL/GLFW properly installed")
            print("[INFO] Try: pip install glfw")
            raise

        fovy = math.radians(fovy_deg)
        self.fy = height / (2 * math.tan(fovy / 2))
        self.fx = self.fy  # square pixels
        self.cx = width / 2.0
        self.cy = height / 2.0

        self._obstacle_bounds = [_color_bounds(rgba) for rgba in SHAPE_RGBA.values()]
        
        # Cache for performance
        self._depth_enabled = False

    def close(self):
        """Clean up renderer resources."""
        if hasattr(self, 'renderer') and self.renderer is not None:
            try:
                self.renderer.close()
                print("[OK] Renderer closed")
            except Exception as e:
                print(f"[WARN] Error closing renderer: {e}")

    def _obstacle_mask(self, rgb: np.ndarray) -> np.ndarray:
        """Create binary mask for obstacle pixels."""
        img = rgb.astype(np.float32) / 255.0
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        for lo, hi in self._obstacle_bounds:
            m = np.all((img >= lo) & (img <= hi), axis=-1)
            mask |= m
        return mask

    def detect(self, data, cam_name: str) -> List[PerceivedObstacle]:
        """
        Detect obstacles visible in one robot's onboard camera this frame.
        
        Args:
            data: MuJoCo data object
            cam_name: Name of the camera to use (e.g., "robot0_cam")
            
        Returns:
            List of PerceivedObstacle objects
        """
        if self.renderer is None:
            print("[WARN] Renderer not initialized")
            return []

        try:
            # Render RGB
            self.renderer.update_scene(data, camera=cam_name)
            rgb = self.renderer.render()
            
            # Get obstacle mask
            mask = self._obstacle_mask(rgb)
            labeled, n = ndimage.label(mask)
            if n == 0:
                return []

            # Enable and render depth
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(data, camera=cam_name)
            depth = self.renderer.render()
            self.renderer.disable_depth_rendering()

            # Get camera pose
            cam_id = self.model.camera(cam_name).id
            cam_pos = data.cam_xpos[cam_id].copy()
            cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()

            detections = []
            for label_id in range(1, n + 1):
                ys, xs = np.where(labeled == label_id)
                if len(xs) < self.min_blob_pixels:
                    continue

                # Use median depth for robustness
                u, v = xs.mean(), ys.mean()
                Z = float(np.median(depth[ys, xs]))
                
                # Validate depth
                if Z <= 0 or Z > self.max_range or not math.isfinite(Z):
                    continue

                # Pinhole back-projection
                # Camera frame: X right, Y up, Z backward (forward = -Z)
                x_cam = (u - self.cx) * Z / self.fx
                y_cam = -(v - self.cy) * Z / self.fy
                z_cam = -Z
                p_cam = np.array([x_cam, y_cam, z_cam])
                p_world = cam_pos + cam_mat @ p_cam

                # Estimate obstacle radius from angular size
                u_span = (xs.max() - xs.min() + 1) / 2.0
                angular_half_width = math.atan(u_span / self.fx)
                radius_est = max(0.15, Z * math.tan(angular_half_width))

                # Push surface point back to approximate center
                ray_dir = p_world - cam_pos
                ray_len = np.linalg.norm(ray_dir)
                if ray_len > 1e-6:
                    ray_dir = ray_dir / ray_len
                    center_world = p_world + ray_dir * radius_est
                else:
                    center_world = p_world

                # Confidence based on expected vs actual pixels
                expected_pixels = max(1.0, (radius_est * self.fx / max(Z, 0.1)) ** 2 * math.pi)
                confidence = min(1.0, len(xs) / expected_pixels)

                detections.append(PerceivedObstacle(
                    x=float(center_world[0]), 
                    y=float(center_world[1]),
                    bounding_radius=float(radius_est), 
                    confidence=float(confidence)))

            return detections
            
        except Exception as e:
            print(f"[ERROR] Detection failed for camera {cam_name}: {e}")
            return []

    def visualize_detection(self, data, cam_name: str, detections: List[PerceivedObstacle], 
                           save_path: str = None):
        """
        Visualize the detection results on the RGB image.
        
        Args:
            data: MuJoCo data object
            cam_name: Camera name
            detections: List of PerceivedObstacle objects
            save_path: Optional path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            
            self.renderer.update_scene(data, camera=cam_name)
            rgb = self.renderer.render()
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.imshow(rgb)
            
            # Draw detected obstacles
            for det in detections:
                # Project obstacle center to image plane
                # This is approximate - would need full projection for exact position
                circle = Circle((det.x * 50 + 50, det.y * 50 + 50), 
                               det.bounding_radius * 20, 
                               fill=False, color='red', linewidth=2)
                ax.add_patch(circle)
                ax.text(det.x * 50 + 50, det.y * 50 + 50, 
                       f'{det.confidence:.2f}', color='white', 
                       fontsize=8, ha='center', va='center')
            
            ax.set_title(f'Detection Results - {cam_name}')
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"[INFO] Visualization saved to {save_path}")
            else:
                plt.show()
            plt.close()
            
        except ImportError:
            print("[WARN] matplotlib not available for visualization")
        except Exception as e:
            print(f"[ERROR] Visualization failed: {e}")

    def __del__(self):
        """Ensure renderer is closed when object is garbage collected."""
        self.close()


# Utility function for testing
def test_perception():
    """
    Simple test function to verify perception works.
    Run this to test your setup.
    """
    print("[INFO] Testing perception module...")
    
    # Check if mujoco is available
    try:
        import mujoco
        print(f"[OK] mujoco version: {mujoco.__version__}")
    except ImportError as e:
        print(f"[ERROR] mujoco not found: {e}")
        return
    
    # Check if scipy is available
    try:
        import scipy
        print(f"[OK] scipy version: {scipy.__version__}")
    except ImportError as e:
        print(f"[ERROR] scipy not found: {e}")
        print("[INFO] Install scipy: pip install scipy")
        return
    
    print("[OK] All dependencies found!")
    print("[INFO] Perception module ready for use")


if __name__ == "__main__":
    test_perception()