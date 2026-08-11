import os
import cv2
import time
import numpy as np
from pathlib import Path

from helpers import *
from display3d_open3d import Display3D

# TODO: odometry poses (steering, speed)
# TODO: IMU poses
# TODO: GNSS poses (translation only?)
# TODO: visual odometry (copy SLAM)
# TODO: sensor fusion with Kalman filter
# TODO: compare error with ground truth + save predicted poses

# env vars
DATA_PATH = os.getenv("DATA_PATH", None)
if DATA_PATH is None:
  print("Usage: DATA_PATH=<path_to_data> python post_process_clip.py")

RENDER = os.getenv("RENDER", 1) # TODO: default 0 when done
ODOMETRY_DT = float(os.getenv("ODOMETRY_DT", 0.05))
ODOMETRY_WHEELBASE_M = float(os.getenv("ODOMETRY_WHEELBASE_M", 2.875))
ODOMETRY_MAX_STEER_DEG = float(os.getenv("ODOMETRY_MAX_STEER_DEG", 35.0))


def load_array(path, mmap_mode=None):
  if not Path(path).exists():
    raise FileNotFoundError(path)
  return np.load(path, mmap_mode=mmap_mode)

def load_data(path):
  video_path = os.path.join(path, "video.hevc")
  steers_path = os.path.join(path, "steering_angles.npy")
  speeds_path = os.path.join(path, "speeds.npy")
  poses_path = os.path.join(path, "poses.npy")
  imu_path = os.path.join(path, "imu.npy")
  gnss_path = os.path.join(path, "gnss.npy")

  return {
    "video_path": video_path,
    "steers": load_array(steers_path),
    "speeds": load_array(speeds_path),
    "poses": load_array(poses_path),
    "imu": load_array(imu_path),
    "gnss": load_array(gnss_path)
  }

def print_frame_data(frame_idx, arrays):
  throttle = arrays.get("throttles")
  throttle_str = "N/A" if throttle is None else f"{float(throttle[frame_idx]):.6f}"
  print(f"[frame {frame_idx}]")
  print(f"pose [x y z roll pitch yaw]: {arrays['poses'][frame_idx]}")
  print(f"steering_angle: {float(arrays['steers'][frame_idx]):.6f}")
  print(f"throttle: {throttle_str}")
  print(f"speed_mps: {float(arrays['speeds'][frame_idx]):.6f}")
  print(f"imu [acc xyz gyro xyz compass]: {arrays['imu'][frame_idx]}")
  print(f"gnss [lat lon alt]: {arrays['gnss'][frame_idx]}")


def pose_from_odometry(curr_pose, steering_angle, speed_mps, dt):
  x, y, z, roll, pitch, yaw_deg = curr_pose
  yaw = np.deg2rad(float(yaw_deg))
  steer = np.clip(float(steering_angle), -1.0, 1.0)
  steering_rad = steer * np.deg2rad(ODOMETRY_MAX_STEER_DEG)
  distance = float(speed_mps) * float(dt)

  if abs(steering_rad) < 1e-6:
    next_x = x + distance * np.cos(yaw)
    next_y = y + distance * np.sin(yaw)
    next_yaw = yaw
  else:
    yaw_rate = float(speed_mps) / ODOMETRY_WHEELBASE_M * np.tan(steering_rad)
    next_yaw = yaw + yaw_rate * float(dt)

    if abs(yaw_rate) < 1e-6:
      next_x = x + distance * np.cos(yaw)
      next_y = y + distance * np.sin(yaw)
    else:
      turn_radius = float(speed_mps) / yaw_rate
      next_x = x + turn_radius * (np.sin(next_yaw) - np.sin(yaw))
      next_y = y - turn_radius * (np.cos(next_yaw) - np.cos(yaw))

  return np.array([
    next_x,
    next_y,
    z,
    roll,
    pitch,
    np.rad2deg(next_yaw),
  ], dtype=np.float32)


def odometry_from_controls(steering_angles, speeds_mps, dt):
  odometry_poses = np.zeros((len(speeds_mps), 6), dtype=np.float32)
  for frame_idx in range(1, len(odometry_poses)):
    odometry_poses[frame_idx] = pose_from_odometry(
      odometry_poses[frame_idx - 1],
      steering_angles[frame_idx - 1],
      speeds_mps[frame_idx - 1],
      dt,
    )
  return odometry_poses


if __name__ == "__main__":
  print(f"[*] Replay data: {DATA_PATH}")
  data = load_data(DATA_PATH)

  display_poses, display_path, pose_scale = normalize_poses(data["poses"])
  print(f"[*] Pose scale: {pose_scale:.6f} display units per CARLA unit")

  odometry_poses = odometry_from_controls(data["steers"], data["speeds"], ODOMETRY_DT)
  odometry_display_poses, odometry_display_path, odometry_pose_scale = normalize_poses(odometry_poses)
  print(f"[*] Odometry pose scale: {odometry_pose_scale:.6f} display units per odometry unit")

  cap = cv2.VideoCapture(data["video_path"])
  if not cap.isOpened():
    raise RuntimeError(f"Could not open HEVC video: {data["video_path"]}")

  total_frames = frame_count(cap, data)
  max_frames = os.getenv("MAX_FRAMES")
  if max_frames is not None:
    total_frames = min(total_frames, int(max_frames))
  lookahead = int(os.getenv("LOOKAHEAD", DEFAULT_LOOKAHEAD))
  print(f"[*] Total replay frames: {total_frames}")

  display_3d = None
  if RENDER:
    display_3d = Display3D(DISPLAY_3D_W, DISPLAY_3D_H, max_frames=total_frames)

  try:
    for frame_idx in range(total_frames):
      ret, frame = cap.read()
      if not ret:
        print(f"[!] Video ended at frame {frame_idx}")
        break

      print_frame_data(frame_idx, data)

      if display_3d is not None:
        display_3d.draw(display_poses, display_path, frame_idx, stream_id="ground_truth")
        display_3d.draw(odometry_display_poses, path=None, frame_idx=frame_idx, color=(0.0, 0.4, 1.0), stream_id="odometry")

      if RENDER:
        cv2.imshow("DISPLAY 2D", frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):
          break

      print()
  finally:
    cap.release()
    cv2.destroyAllWindows()
