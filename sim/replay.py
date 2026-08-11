#!/usr/bin/env python3
import math
import os
from pathlib import Path

import cv2
import numpy as np

from display3d import Display3D
from helpers import *


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = (SCRIPT_DIR.parent / "data" / "1").resolve()

# TODO: 3D display camera should follow current pose + draw only current pose frustum


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

def load_optional_array(data_path, filename, mmap_mode=None):
  path = data_path / filename
  if not path.exists():
    return None
  return np.load(path, mmap_mode=mmap_mode)


def load_dataset(data_path):
  return {
    "poses": load_required_array(data_path, "poses.npy"),
    "predicted_poses": load_optional_array(data_path, "predicted_poses.npy"),
    "desires": load_required_array(data_path, "desires.npy"),
    "steering_angles": load_required_array(data_path, "steering_angles.npy"),
    "throttles": load_required_array(data_path, "throttles.npy"),
    "speeds": load_required_array(data_path, "speeds.npy"),
    "imu": load_required_array(data_path, "imu.npy"),
    "gnss": load_required_array(data_path, "gnss.npy"),
    "segmentation": load_required_array(data_path, "segmentation.npy", mmap_mode="r"),
  }


def pose_stream_from_zero_start(local_poses, reference_poses):
  local_poses = np.asarray(local_poses, dtype=np.float32)
  reference_poses = np.asarray(reference_poses, dtype=np.float32)
  raw_poses = local_poses.copy()
  initial_rot = euler_to_rotation_matrix(
    reference_poses[0, 3],
    reference_poses[0, 4],
    reference_poses[0, 5],
  )
  raw_poses[:, :3] = local_poses[:, :3] @ initial_rot.T + reference_poses[0, :3]
  raw_poses[:, 3] = local_poses[:, 3] + reference_poses[0, 3]
  raw_poses[:, 4] = local_poses[:, 4] + reference_poses[0, 4]
  raw_poses[:, 5] = local_poses[:, 5] + reference_poses[0, 5]
  return raw_poses


def display_poses_from_reference(raw_poses, reference_pose, scale):
  reference_position = reference_pose[:3].astype(np.float64)
  positions = raw_poses[:, :3].astype(np.float64)
  relative = positions - reference_position
  display_path = np.column_stack((relative[:, 0], relative[:, 2], -relative[:, 1])) * float(scale)

  display_poses = []
  for pose_row, position in zip(raw_poses, display_path):
    transform = np.eye(4)
    transform[:3, :3] = euler_to_rotation_matrix(pose_row[3], pose_row[4], pose_row[5])
    transform[:3, 3] = position
    display_poses.append(transform)
  return np.asarray(display_poses), display_path


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


def draw_projected_path(frame, image_points, line_color=(0, 0, 255), current_color=None):
  if len(image_points) == 0:
    return frame
  if current_color is None:
    current_color = line_color

  if len(image_points) > 1:
    cv2.polylines(frame, [image_points.reshape((-1, 1, 2))], False, line_color, 3)
  for idx, point in enumerate(image_points):
    radius = 3 if idx < len(image_points) - 1 else 6
    color = line_color if idx < len(image_points) - 1 else current_color
    cv2.circle(frame, tuple(point), radius, color, -1)
  return frame


def print_frame_data(frame_idx, arrays):
  desire = int(arrays["desires"][frame_idx])
  throttle = arrays.get("throttles")
  throttle_str = "N/A" if throttle is None else f"{float(throttle[frame_idx]):.6f}"
  print(f"[frame {frame_idx}]")
  print(f"pose [x y z roll pitch yaw]: {arrays['poses'][frame_idx]}")
  print(f"desire: {desire} ({DESIRE.get(desire, 'unknown')})")
  print(f"steering_angle: {float(arrays['steering_angles'][frame_idx]):.6f}")
  print(f"throttle: {throttle_str}")
  print(f"speed_mps: {float(arrays['speeds'][frame_idx]):.6f}")
  print(f"imu [acc xyz gyro xyz compass]: {arrays['imu'][frame_idx]}")
  print(f"gnss [lat lon alt]: {arrays['gnss'][frame_idx]}")
  labels = arrays["segmentation"][frame_idx]
  unique_labels = np.unique(labels)
  print(f"segmentation labels: {unique_labels.tolist()}")


def annotate_frame(frame, frame_idx, total_frames, arrays, pose_scale):
  desire = int(arrays["desires"][frame_idx])
  throttle = arrays.get("throttles")
  throttle_text = "N/A" if throttle is None else f"{float(throttle[frame_idx]):.3f}"
  text = [
    f"frame {frame_idx + 1}/{total_frames}",
    f"desire: {DESIRE.get(desire, 'unknown')}",
    f"speed: {float(arrays['speeds'][frame_idx]):.2f} m/s",
    f"steer: {float(arrays['steering_angles'][frame_idx]):.3f}",
    f"throttle: {throttle_text}",
    f"pose scale: {pose_scale:.4f}",
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
  ground_truth_poses = arrays["poses"]
  predicted_world_poses = None
  if arrays["predicted_poses"] is not None:
    predicted_count = min(len(arrays["predicted_poses"]), len(ground_truth_poses))
    predicted_world_poses = pose_stream_from_zero_start(
      arrays["predicted_poses"][:predicted_count],
      ground_truth_poses[:predicted_count],
    )

  display_poses, display_path, pose_scale = normalize_poses(ground_truth_poses)
  print(f"[*] Pose scale: {pose_scale:.6f} display units per CARLA unit")
  predicted_display_poses = None
  predicted_display_path = None
  if predicted_world_poses is not None:
    predicted_display_poses, predicted_display_path = display_poses_from_reference(
      predicted_world_poses,
      ground_truth_poses[0],
      pose_scale,
    )
    print("[*] Rendering predicted poses: yes")
  else:
    print("[*] Rendering predicted poses: no")

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
        display_3d.draw(display_poses, display_path, frame_idx, stream_id="ground_truth")
        if predicted_display_poses is not None and frame_idx < len(predicted_display_poses):
          display_3d.draw(
            predicted_display_poses,
            predicted_display_path,
            frame_idx,
            color=(1.0, 0.0, 0.0),
            stream_id="predicted",
          )

      if render_2d:
        gt_points = project_path_to_image(ground_truth_poses, frame_idx, lookahead, frame.shape)
        frame = draw_projected_path(frame, gt_points, line_color=(0, 255, 0), current_color=(0, 180, 0))
        if predicted_world_poses is not None and frame_idx < len(predicted_world_poses):
          predicted_points = project_path_to_image(predicted_world_poses, frame_idx, lookahead, frame.shape)
          frame = draw_projected_path(frame, predicted_points, line_color=(0, 0, 255), current_color=(0, 0, 180))
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
