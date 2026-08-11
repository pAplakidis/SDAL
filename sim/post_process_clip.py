import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from carla_config import FIXED_DELTA_SECONDS, IMG_WIDTH, IMG_HEIGHT
from helpers import *
from display3d import Display3D
from utils.coordinates import LocalCoord
from visual_odometry import VisualOdometry, draw_keypoints, pose_to_display_transform

# TODO: sensor fusion with Kalman filter
# TODO: time for performance and latency analysis
# TODO: compare error with ground truth + save predicted poses

DATA_PATH = os.getenv("DATA_PATH", None)
if DATA_PATH is None:
  print("Usage: DATA_PATH=<path_to_data> python post_process_clip.py")

RENDER = os.getenv("RENDER", "1").lower() not in ("0", "false", "no", "off", "")

# Edit these constants directly when tuning post-processing.
ODOMETRY_DT = FIXED_DELTA_SECONDS
ODOMETRY_WHEELBASE_M = 2.875
ODOMETRY_MAX_STEER_DEG = 35.0
IMU_GRAVITY_MPS2 = 9.81
IMU_ACCEL_MAX_MPS2 = 100.0
VO_ENABLED = True
VO_IMAGE_SCALE = 0.5
VO_MAX_FEATURES = 2000
VO_MATCH_RATIO = 0.75
VO_MAX_HAMMING_DISTANCE = 32
VO_MIN_INLIERS = 20
VO_RANSAC_THRESHOLD = 1.0


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


def wrap_angle_deg(angle_deg):
  return (float(angle_deg) + 180.0) % 360.0 - 180.0


def zero_start_pose_stream(raw_poses):
  raw_poses = np.asarray(raw_poses, dtype=np.float32)
  zeroed = raw_poses.copy()
  initial_rot = euler_to_rotation_matrix(zeroed[0, 3], zeroed[0, 4], zeroed[0, 5])
  zeroed[:, :3] = (zeroed[:, :3] - zeroed[0, :3]) @ initial_rot
  zeroed[:, 3] = np.vectorize(wrap_angle_deg)(zeroed[:, 3] - zeroed[0, 3])
  zeroed[:, 4] = np.vectorize(wrap_angle_deg)(zeroed[:, 4] - zeroed[0, 4])
  zeroed[:, 5] = np.vectorize(wrap_angle_deg)(zeroed[:, 5] - zeroed[0, 5])
  return zeroed


def gnss_poses_from_measurements(gnss_data, reference_pose):
  gnss_data = np.asarray(gnss_data, dtype=np.float64)
  converter = LocalCoord.from_geodetic(gnss_data[0])
  ned_positions = converter.geodetic2ned(gnss_data)
  carla_positions = np.column_stack((ned_positions[:, 1], ned_positions[:, 0], -ned_positions[:, 2]))

  gnss_poses = np.zeros((len(gnss_data), 6), dtype=np.float32)
  gnss_poses[:, :3] = carla_positions.astype(np.float32)
  gnss_poses[:, 3] = float(reference_pose[3])
  gnss_poses[:, 4] = float(reference_pose[4])
  gnss_poses[:, 5] = float(reference_pose[5])
  return gnss_poses


# assumes bicycle model kinematics
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


def imu_pose_from_measurement(curr_pose, curr_velocity, imu, dt):
  curr_pose = np.asarray(curr_pose, dtype=np.float64)
  curr_velocity = np.asarray(curr_velocity, dtype=np.float64)

  roll_deg, pitch_deg, yaw_deg = curr_pose[3:6]
  gyro = np.asarray([
    imu[3],
    imu[4],
    imu[5],
  ], dtype=np.float64)
  accel_body = np.asarray([
    imu[0],
    imu[1],
    imu[2],
  ], dtype=np.float64)

  if np.linalg.norm(accel_body) > IMU_ACCEL_MAX_MPS2:
    accel_body = np.zeros(3, dtype=np.float64)

  roll_deg = wrap_angle_deg(roll_deg + np.rad2deg(gyro[0] * dt))
  pitch_deg = wrap_angle_deg(pitch_deg + np.rad2deg(gyro[1] * dt))
  yaw_deg = wrap_angle_deg(yaw_deg + np.rad2deg(gyro[2] * dt))

  world_accel = euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg).dot(accel_body)
  world_accel[2] -= IMU_GRAVITY_MPS2
  next_velocity = curr_velocity + world_accel * dt
  next_position = curr_pose[:3] + next_velocity * dt

  return (
    np.array([
      next_position[0],
      next_position[1],
      next_position[2],
      roll_deg,
      pitch_deg,
      yaw_deg,
    ], dtype=np.float32),
    next_velocity,
  )


def imu_poses_from_measurements(imu_data, dt):
  imu_poses = np.zeros((len(imu_data), 6), dtype=np.float32)
  velocity = np.zeros(3, dtype=np.float64)
  for frame_idx in range(1, len(imu_poses)):
    imu_poses[frame_idx], velocity = imu_pose_from_measurement(
      imu_poses[frame_idx - 1],
      velocity,
      imu_data[frame_idx],
      dt,
    )
  return imu_poses


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

  cap = cv2.VideoCapture(data["video_path"])
  if not cap.isOpened():
    raise RuntimeError(f"Could not open HEVC video: {data['video_path']}")

  total_frames = frame_count(cap, data)
  max_frames = os.getenv("MAX_FRAMES")
  if max_frames is not None:
    total_frames = min(total_frames, int(max_frames))
  lookahead = int(os.getenv("LOOKAHEAD", DEFAULT_LOOKAHEAD))
  print(f"[*] Total replay frames: {total_frames}")

  display_poses, display_path, pose_scale = normalize_poses(zero_start_pose_stream(data["poses"]))
  print(f"[*] Pose scale: {pose_scale:.6f} display units per CARLA unit")

  odometry_poses = zero_start_pose_stream(odometry_from_controls(data["steers"], data["speeds"], ODOMETRY_DT))
  odometry_display_poses, odometry_display_path, odometry_pose_scale = normalize_poses(odometry_poses)
  print(f"[*] Odometry pose scale: {odometry_pose_scale:.6f} display units per odometry unit")

  imu_poses = zero_start_pose_stream(imu_poses_from_measurements(data["imu"], ODOMETRY_DT))
  imu_display_poses, imu_display_path, imu_pose_scale = normalize_poses(imu_poses)
  print(f"[*] IMU pose scale: {imu_pose_scale:.6f} display units per IMU unit")

  gnss_poses = zero_start_pose_stream(gnss_poses_from_measurements(data["gnss"], data["poses"][0]))
  gnss_display_poses, gnss_display_path, gnss_pose_scale = normalize_poses(gnss_poses)
  print(f"[*] GNSS pose scale: {gnss_pose_scale:.6f} display units per GNSS unit")

  vo_display_poses = []
  if VO_ENABLED:
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if video_width <= 0 or video_height <= 0:
      video_width, video_height = IMG_WIDTH, IMG_HEIGHT
    fx, fy, cx, cy = camera_intrinsics(video_width, video_height, CAMERA_FOV_DEG)
    camera_matrix = np.array([
      [fx, 0.0, cx],
      [0.0, fy, cy],
      [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    vo = VisualOdometry(
      camera_matrix,
      image_scale=VO_IMAGE_SCALE,
      max_features=VO_MAX_FEATURES,
      ratio=VO_MATCH_RATIO,
      max_hamming_distance=VO_MAX_HAMMING_DISTANCE,
      min_inliers=VO_MIN_INLIERS,
      ransac_threshold=VO_RANSAC_THRESHOLD,
    )
    print("[*] Visual odometry enabled")

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

      if VO_ENABLED:
        speed = None if frame_idx == 0 else data["speeds"][frame_idx - 1]
        vo_pose, vo_debug = vo.step(frame, speed_mps=speed, dt=ODOMETRY_DT)
        vo_display_poses.append(pose_to_display_transform(vo_pose, pose_scale))
        frame = draw_keypoints(frame, vo_debug["keypoints"], color=(0, 255, 255), radius=2)

      if display_3d is not None:
        display_3d.draw(display_poses, display_path, frame_idx, stream_id="ground_truth")
        display_3d.draw(odometry_display_poses, path=None, frame_idx=frame_idx, color=(0.0, 0.4, 1.0), stream_id="odometry")
        display_3d.draw(imu_display_poses, path=None, frame_idx=frame_idx, color=(1.0, 0.0, 1.0), stream_id="imu")
        display_3d.draw(gnss_display_poses, path=None, frame_idx=frame_idx, color=(1.0, 1.0, 0.0), stream_id="gnss")
        if vo_display_poses:
          display_3d.draw(np.asarray(vo_display_poses), path=None, frame_idx=frame_idx, color=(0.0, 1.0, 1.0), stream_id="visual_odometry")

      if RENDER:
        cv2.imshow("DISPLAY 2D", frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):
          break

      print()
  finally:
    cap.release()
    cv2.destroyAllWindows()
