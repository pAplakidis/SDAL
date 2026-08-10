#!/usr/bin/env python3
import math
import os
from pathlib import Path

import cv2
import numpy as np

from display3d_open3d import Display3D


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = (SCRIPT_DIR.parent / "data" / "1").resolve()

DISPLAY_3D_W = 960
DISPLAY_3D_H = 540
TARGET_PATH_EXTENT = 50.0
DEFAULT_LOOKAHEAD = 200
CAMERA_FOV_DEG = 70.0
CAMERA_OFFSET_CARLA = np.array([0.8, 0.0, 1.13], dtype=np.float64)
DESIRE = {
  0: "forward",
  1: "right",
  2: "left",
}

# CARLA semantic segmentation label colors, BGR for OpenCV display.
CITYSCAPES_PALETTE_BGR = np.array([
  [0, 0, 0],        # 0 Unlabeled
  [70, 70, 70],     # 1 Building
  [100, 40, 40],    # 2 Fence
  [55, 90, 80],     # 3 Other
  [220, 20, 60],    # 4 Pedestrian
  [153, 153, 153],  # 5 Pole
  [157, 234, 50],   # 6 RoadLine
  [128, 64, 128],   # 7 Road
  [232, 35, 244],   # 8 SideWalk
  [35, 142, 107],   # 9 Vegetation
  [142, 0, 0],      # 10 Vehicles
  [156, 102, 102],  # 11 Wall
  [180, 130, 70],   # 12 TrafficSign
  [81, 0, 81],      # 13 Sky
  [230, 150, 140],  # 14 Ground
  [180, 165, 180],  # 15 Bridge
  [250, 170, 30],   # 16 RailTrack
  [110, 190, 160],  # 17 GuardRail
  [170, 120, 50],   # 18 TrafficLight
  [45, 60, 150],    # 19 Static
  [145, 170, 100],  # 20 Dynamic
  [150, 100, 100],  # 21 Water
  [0, 0, 230],      # 22 Terrain
  [110, 80, 100],   # 23 Any
  [0, 0, 142],      # 24 Car
  [0, 0, 70],       # 25 Truck
  [0, 60, 100],     # 26 Bus
  [0, 0, 90],       # 27 Train
  [0, 0, 110],      # 28 Motorcycle
  [0, 80, 100],     # 29 Bicycle
], dtype=np.uint8)


def env_bool(name, default=True):
  value = os.getenv(name)
  if value is None:
    return default
  return value.lower() not in ("0", "false", "no", "off")


def resolve_data_path():
  return Path(os.getenv("DATA_PATH", str(DEFAULT_DATA_PATH))).expanduser().resolve()


def load_required_array(data_path, filename, mmap_mode=None):
  path = data_path / filename
  if not path.exists():
    raise FileNotFoundError(path)
  return np.load(path, mmap_mode=mmap_mode)


def load_dataset(data_path):
  return {
    "poses": load_required_array(data_path, "poses.npy"),
    "desires": load_required_array(data_path, "desires.npy"),
    "steering_angles": load_required_array(data_path, "steering_angles.npy"),
    "speeds": load_required_array(data_path, "speeds.npy"),
    "imu": load_required_array(data_path, "imu.npy"),
    "gnss": load_required_array(data_path, "gnss.npy"),
    "segmentation": load_required_array(data_path, "segmentation.npy", mmap_mode="r"),
  }


def euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
  roll = math.radians(float(roll_deg))
  pitch = math.radians(float(pitch_deg))
  yaw = math.radians(float(yaw_deg))

  cr, sr = math.cos(roll), math.sin(roll)
  cp, sp = math.cos(pitch), math.sin(pitch)
  cy, sy = math.cos(yaw), math.sin(yaw)

  rx = np.array([
    [1.0, 0.0, 0.0],
    [0.0, cr, -sr],
    [0.0, sr, cr],
  ])
  ry = np.array([
    [cp, 0.0, sp],
    [0.0, 1.0, 0.0],
    [-sp, 0.0, cp],
  ])
  rz = np.array([
    [cy, -sy, 0.0],
    [sy, cy, 0.0],
    [0.0, 0.0, 1.0],
  ])
  return rz @ ry @ rx


def normalize_poses(raw_poses, target_extent=TARGET_PATH_EXTENT):
  positions = raw_poses[:, :3].astype(np.float64)
  relative = positions - positions[0]

  # CARLA/UE x,y,z -> display x,z,y. Invert y so forward motion looks right-handed.
  display_path = np.column_stack((relative[:, 0], relative[:, 2], -relative[:, 1]))
  extent = np.ptp(display_path, axis=0).max()
  scale = 1.0 if extent <= 0 else target_extent / extent
  display_path *= scale

  display_poses = []
  for pose_row, position in zip(raw_poses, display_path):
    transform = np.eye(4)
    transform[:3, :3] = euler_to_rotation_matrix(pose_row[3], pose_row[4], pose_row[5])
    transform[:3, 3] = position
    display_poses.append(transform)

  return np.asarray(display_poses), display_path, scale


def colorize_segmentation(labels):
  labels = np.asarray(labels, dtype=np.uint8)
  output = np.zeros(labels.shape + (3,), dtype=np.uint8)
  known = labels < len(CITYSCAPES_PALETTE_BGR)
  output[known] = CITYSCAPES_PALETTE_BGR[labels[known]]
  if np.any(~known):
    unknown = labels[~known]
    output[~known] = np.stack(
      ((unknown * 37) % 255, (unknown * 17) % 255, (unknown * 97) % 255),
      axis=1,
    )
  return output


def camera_intrinsics(width, height, fov_deg):
  focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
  return focal, focal, width / 2.0, height / 2.0


def project_path_to_image(raw_poses, frame_idx, lookahead, image_shape):
  height, width = image_shape[:2]
  end_idx = min(len(raw_poses), frame_idx + lookahead)
  if end_idx <= frame_idx:
    return np.zeros((0, 2), dtype=np.int32)

  pose = raw_poses[frame_idx]
  vehicle_position = pose[:3].astype(np.float64)
  vehicle_rotation = euler_to_rotation_matrix(pose[3], pose[4], pose[5])
  camera_position = vehicle_position + vehicle_rotation @ CAMERA_OFFSET_CARLA

  path_world = raw_poses[frame_idx:end_idx, :3].astype(np.float64)
  path_camera = (path_world - camera_position) @ vehicle_rotation

  depth = path_camera[:, 0]
  valid = depth > 0.1
  if not np.any(valid):
    return np.zeros((0, 2), dtype=np.int32)

  path_camera = path_camera[valid]
  depth = depth[valid]
  fx, fy, cx, cy = camera_intrinsics(width, height, CAMERA_FOV_DEG)
  image_points = np.column_stack((
    fx * (path_camera[:, 1] / depth) + cx,
    cy - fy * (path_camera[:, 2] / depth),
  ))

  finite = np.isfinite(image_points).all(axis=1)
  in_frame = (
    finite
    & (image_points[:, 0] >= 0)
    & (image_points[:, 0] < width)
    & (image_points[:, 1] >= 0)
    & (image_points[:, 1] < height)
  )
  return image_points[in_frame].astype(np.int32)


def draw_projected_path(frame, image_points):
  if len(image_points) == 0:
    return frame

  if len(image_points) > 1:
    cv2.polylines(frame, [image_points.reshape((-1, 1, 2))], False, (0, 0, 255), 3)
  for idx, point in enumerate(image_points):
    radius = 3 if idx < len(image_points) - 1 else 6
    color = (0, 0, 255) if idx < len(image_points) - 1 else (0, 0, 180)
    cv2.circle(frame, tuple(point), radius, color, -1)
  return frame


def frame_count(cap, arrays):
  counts = [len(value) for value in arrays.values() if hasattr(value, "__len__")]
  video_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if video_count > 0:
    counts.append(video_count)
  return min(counts)


def print_frame_data(frame_idx, arrays):
  desire = int(arrays["desires"][frame_idx])
  print(f"[frame {frame_idx}]")
  print(f"pose [x y z roll pitch yaw]: {arrays['poses'][frame_idx]}")
  print(f"desire: {desire} ({DESIRE.get(desire, 'unknown')})")
  print(f"steering_angle: {float(arrays['steering_angles'][frame_idx]):.6f}")
  print(f"speed_mps: {float(arrays['speeds'][frame_idx]):.6f}")
  print(f"imu [acc xyz gyro xyz compass]: {arrays['imu'][frame_idx]}")
  print(f"gnss [lat lon alt]: {arrays['gnss'][frame_idx]}")
  labels = arrays["segmentation"][frame_idx]
  unique_labels = np.unique(labels)
  print(f"segmentation labels: {unique_labels.tolist()}")


def annotate_frame(frame, frame_idx, total_frames, arrays, pose_scale):
  desire = int(arrays["desires"][frame_idx])
  text = [
    f"frame {frame_idx + 1}/{total_frames}",
    f"desire: {DESIRE.get(desire, 'unknown')}",
    f"speed: {float(arrays['speeds'][frame_idx]):.2f} m/s",
    f"steer: {float(arrays['steering_angles'][frame_idx]):.3f}",
  ]
  for i, line in enumerate(text):
    cv2.putText(
      frame,
      line,
      (24, 36 + i * 32),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (0, 255, 0),
      2,
      cv2.LINE_AA,
    )
  return frame


if __name__ == "__main__":
  data_path = resolve_data_path()
  video_path = data_path / "video.hevc"
  if not video_path.exists():
    raise FileNotFoundError(video_path)

  print(f"[*] Replay data: {data_path}")
  arrays = load_dataset(data_path)
  display_poses, display_path, pose_scale = normalize_poses(arrays["poses"])
  print(f"[*] Pose scale: {pose_scale:.6f} display units per CARLA unit")

  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    raise RuntimeError(f"Could not open HEVC video: {video_path}")

  total_frames = frame_count(cap, arrays)
  max_frames = os.getenv("MAX_FRAMES")
  if max_frames is not None:
    total_frames = min(total_frames, int(max_frames))
  lookahead = int(os.getenv("LOOKAHEAD", DEFAULT_LOOKAHEAD))
  print(f"[*] Total replay frames: {total_frames}")

  render_3d = env_bool("RENDER_3D", True)
  render_2d = env_bool("RENDER_2D", True)
  display_3d = None
  if render_3d:
    display_3d = Display3D(DISPLAY_3D_W, DISPLAY_3D_H, max_frames=total_frames)

  try:
    for frame_idx in range(total_frames):
      ret, frame = cap.read()
      if not ret:
        print(f"[!] Video ended at frame {frame_idx}")
        break

      print_frame_data(frame_idx, arrays)

      if display_3d is not None:
        display_3d.draw(display_poses, display_path, frame_idx)

      if render_2d:
        image_points = project_path_to_image(arrays["poses"], frame_idx, lookahead, frame.shape)
        frame = draw_projected_path(frame, image_points)
        frame = annotate_frame(frame, frame_idx, total_frames, arrays, pose_scale)
        segmentation = colorize_segmentation(arrays["segmentation"][frame_idx])

        cv2.imshow("CARLA HEVC", frame)
        cv2.imshow("CARLA Segmentation", segmentation)
        if cv2.waitKey(1) & 0xFF == ord("q"):
          break

      print()
  finally:
    cap.release()
    if render_2d:
      cv2.destroyAllWindows()
    if display_3d is not None:
      display_3d.close()
