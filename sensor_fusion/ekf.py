import math

import numpy as np


def wrap_angle_rad(angle_rad):
  return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def wrap_angle_deg(angle_deg):
  return (float(angle_deg) + 180.0) % 360.0 - 180.0


def euler_to_rotation_matrix_rad(roll, pitch, yaw):
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


def pose_to_measurement(pose_row):
  pose_row = np.asarray(pose_row, dtype=np.float64)
  return np.array([
    pose_row[0],
    pose_row[1],
    pose_row[2],
    math.radians(float(pose_row[3])),
    math.radians(float(pose_row[4])),
    math.radians(float(pose_row[5])),
  ], dtype=np.float64)


def pose_to_display_transform(pose_row, scale):
  pose_row = np.asarray(pose_row, dtype=np.float64)
  transform = np.eye(4, dtype=np.float64)
  transform[:3, :3] = euler_to_rotation_matrix_rad(
    math.radians(float(pose_row[3])),
    math.radians(float(pose_row[4])),
    math.radians(float(pose_row[5])),
  )
  transform[:3, 3] = np.array([
    pose_row[0],
    pose_row[2],
    -pose_row[1],
  ], dtype=np.float64) * float(scale)
  return transform


class PoseEKF:
  def __init__(
    self,
    dt,
    wheelbase_m,
    max_steer_deg,
    gravity_mps2,
    accel_max_mps2,
  ):
    self.dt = float(dt)
    self.wheelbase_m = float(wheelbase_m)
    self.max_steer_rad = math.radians(float(max_steer_deg))
    self.gravity_mps2 = float(gravity_mps2)
    self.accel_max_mps2 = float(accel_max_mps2)

    self.x = np.zeros(9, dtype=np.float64)
    self.P = np.diag([
      1.0, 1.0, 1.0,
      math.radians(5.0) ** 2,
      math.radians(5.0) ** 2,
      math.radians(5.0) ** 2,
      1.0, 1.0, 1.0,
    ])
    self.Q = np.diag([
      0.10 ** 2, 0.10 ** 2, 0.15 ** 2,
      math.radians(0.5) ** 2,
      math.radians(0.5) ** 2,
      math.radians(0.5) ** 2,
      0.30 ** 2, 0.30 ** 2, 0.50 ** 2,
    ])
    self.R_gnss = np.diag([2.0 ** 2, 2.0 ** 2, 4.0 ** 2])
    self.R_vo = np.diag([
      1.5 ** 2, 1.5 ** 2, 2.0 ** 2,
      math.radians(8.0) ** 2,
      math.radians(8.0) ** 2,
      math.radians(8.0) ** 2,
    ])

  def _wrap_state_angles(self):
    self.x[3] = wrap_angle_rad(self.x[3])
    self.x[4] = wrap_angle_rad(self.x[4])
    self.x[5] = wrap_angle_rad(self.x[5])

  def predict(self, steering_angle, speed_mps, imu_row):
    dt = self.dt
    speed = max(float(speed_mps), 0.0)
    steer = np.clip(float(steering_angle), -1.0, 1.0) * self.max_steer_rad
    yaw = self.x[5]

    imu_row = np.asarray(imu_row, dtype=np.float64)
    accel_body = imu_row[:3]
    gyro = imu_row[3:6]
    if np.linalg.norm(accel_body) > self.accel_max_mps2:
      accel_body = np.zeros(3, dtype=np.float64)

    odom_yaw_rate = 0.0
    if abs(steer) > 1e-6:
      odom_yaw_rate = speed / self.wheelbase_m * math.tan(steer)
    yaw_rate = 0.5 * (odom_yaw_rate + float(gyro[2]))

    F = np.eye(9, dtype=np.float64)
    F[0, 5] = -speed * math.sin(yaw) * dt
    F[1, 5] = speed * math.cos(yaw) * dt
    F[2, 8] = dt

    self.x[0] += speed * math.cos(yaw) * dt
    self.x[1] += speed * math.sin(yaw) * dt
    self.x[2] += self.x[8] * dt
    self.x[3] += float(gyro[0]) * dt
    self.x[4] += float(gyro[1]) * dt
    self.x[5] += yaw_rate * dt

    world_accel = euler_to_rotation_matrix_rad(self.x[3], self.x[4], self.x[5]).dot(accel_body)
    world_accel[2] -= self.gravity_mps2
    self.x[6:9] += world_accel * dt
    self.x[6] = speed * math.cos(self.x[5])
    self.x[7] = speed * math.sin(self.x[5])

    self._wrap_state_angles()
    self.P = F @ self.P @ F.T + self.Q

  def _update(self, measurement, H, R, angle_measurement_indices=()):
    z = np.asarray(measurement, dtype=np.float64)
    y = z - H @ self.x
    for measurement_idx in angle_measurement_indices:
      y[measurement_idx] = wrap_angle_rad(y[measurement_idx])

    S = H @ self.P @ H.T + R
    K = self.P @ H.T @ np.linalg.inv(S)
    self.x = self.x + K @ y
    I = np.eye(len(self.x), dtype=np.float64)
    self.P = (I - K @ H) @ self.P
    self._wrap_state_angles()

  def update_gnss(self, gnss_pose):
    H = np.zeros((3, 9), dtype=np.float64)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    self._update(np.asarray(gnss_pose[:3], dtype=np.float64), H, self.R_gnss)

  def update_visual_odometry(self, vo_pose):
    H = np.zeros((6, 9), dtype=np.float64)
    for i in range(6):
      H[i, i] = 1.0
    self._update(pose_to_measurement(vo_pose), H, self.R_vo, angle_measurement_indices=(3, 4, 5))

  def pose(self):
    return np.array([
      self.x[0],
      self.x[1],
      self.x[2],
      math.degrees(self.x[3]),
      math.degrees(self.x[4]),
      math.degrees(self.x[5]),
    ], dtype=np.float32)


def compute_pose_metrics(predicted_poses, ground_truth_poses):
  predicted = np.asarray(predicted_poses, dtype=np.float64)
  ground_truth = np.asarray(ground_truth_poses, dtype=np.float64)
  count = min(len(predicted), len(ground_truth))
  if count == 0:
    return {"frames": 0}
  predicted = predicted[:count]
  ground_truth = ground_truth[:count]

  translation_delta = predicted[:, :3] - ground_truth[:, :3]
  translation_errors = np.linalg.norm(translation_delta, axis=1)
  orientation_delta = np.vectorize(wrap_angle_deg)(predicted[:, 3:6] - ground_truth[:, 3:6])

  return {
    "frames": int(count),
    "translation_rmse_m": float(np.sqrt(np.mean(translation_errors ** 2))),
    "translation_mae_m": float(np.mean(translation_errors)),
    "translation_max_m": float(np.max(translation_errors)),
    "final_translation_error_m": float(translation_errors[-1]),
    "x_rmse_m": float(np.sqrt(np.mean(translation_delta[:, 0] ** 2))),
    "y_rmse_m": float(np.sqrt(np.mean(translation_delta[:, 1] ** 2))),
    "z_rmse_m": float(np.sqrt(np.mean(translation_delta[:, 2] ** 2))),
    "roll_rmse_deg": float(np.sqrt(np.mean(orientation_delta[:, 0] ** 2))),
    "pitch_rmse_deg": float(np.sqrt(np.mean(orientation_delta[:, 1] ** 2))),
    "yaw_rmse_deg": float(np.sqrt(np.mean(orientation_delta[:, 2] ** 2))),
  }
