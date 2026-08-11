import cv2
import numpy as np

from sim.helpers import euler_to_rotation_matrix


CV_TO_LOCAL = np.array([
  [0.0, 0.0, 1.0],
  [1.0, 0.0, 0.0],
  [0.0, -1.0, 0.0],
], dtype=np.float64)


def wrap_angle_deg(angle_deg):
  return (float(angle_deg) + 180.0) % 360.0 - 180.0


def scale_camera_matrix(camera_matrix, image_scale):
  scaled = np.asarray(camera_matrix, dtype=np.float64).copy()
  scaled[0, 0] *= image_scale
  scaled[0, 2] *= image_scale
  scaled[1, 1] *= image_scale
  scaled[1, 2] *= image_scale
  return scaled


def rotation_matrix_to_euler_deg(rotation):
  rotation = np.asarray(rotation, dtype=np.float64)
  sy = -rotation[2, 0]
  sy = np.clip(sy, -1.0, 1.0)
  pitch = np.arcsin(sy)

  if abs(np.cos(pitch)) > 1e-6:
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
  else:
    roll = 0.0
    yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])

  return np.array([
    wrap_angle_deg(np.rad2deg(roll)),
    wrap_angle_deg(np.rad2deg(pitch)),
    wrap_angle_deg(np.rad2deg(yaw)),
  ], dtype=np.float32)


def pose_row_from_cv_pose(position_cv, rotation_cv):
  position_local = CV_TO_LOCAL.dot(position_cv)
  rotation_local = CV_TO_LOCAL.dot(rotation_cv).dot(CV_TO_LOCAL.T)
  euler_deg = rotation_matrix_to_euler_deg(rotation_local)

  pose = np.zeros(6, dtype=np.float32)
  pose[:3] = position_local.astype(np.float32)
  pose[3:] = euler_deg
  return pose


def extract_features(gray, orb, max_features):
  points = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=max_features,
    qualityLevel=0.01,
    minDistance=7,
  )
  if points is None:
    return np.empty((0, 2), dtype=np.float32), None

  keypoints = [cv2.KeyPoint(x=float(point[0][0]), y=float(point[0][1]), size=20) for point in points]
  keypoints, descriptors = orb.compute(gray, keypoints)
  if descriptors is None or keypoints is None:
    return np.empty((0, 2), dtype=np.float32), None

  keypoints = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)
  return keypoints, descriptors


def match_features(prev_descriptors, curr_descriptors, ratio, max_hamming_distance):
  if prev_descriptors is None or curr_descriptors is None:
    return []
  if len(prev_descriptors) < 2 or len(curr_descriptors) < 2:
    return []

  matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
  raw_matches = matcher.knnMatch(prev_descriptors, curr_descriptors, k=2)

  matches = []
  used_prev = set()
  used_curr = set()
  for pair in raw_matches:
    if len(pair) < 2:
      continue
    match, neighbor = pair
    if match.distance >= ratio * neighbor.distance:
      continue
    if match.distance >= max_hamming_distance:
      continue
    if match.queryIdx in used_prev or match.trainIdx in used_curr:
      continue
    used_prev.add(match.queryIdx)
    used_curr.add(match.trainIdx)
    matches.append(match)

  return matches


def recover_relative_pose(prev_keypoints, curr_keypoints, matches, camera_matrix, ransac_threshold):
  if len(matches) < 8:
    return None, None, 0, None

  prev_points = np.asarray([prev_keypoints[match.queryIdx] for match in matches], dtype=np.float32)
  curr_points = np.asarray([curr_keypoints[match.trainIdx] for match in matches], dtype=np.float32)

  essential, mask = cv2.findEssentialMat(
    prev_points,
    curr_points,
    camera_matrix,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=ransac_threshold,
  )
  if essential is None:
    return None, None, 0, None
  if essential.shape[0] > 3:
    essential = essential[:3]

  inlier_count, rotation, translation, pose_mask = cv2.recoverPose(
    essential,
    prev_points,
    curr_points,
    camera_matrix,
    mask=mask,
  )
  inlier_mask = None if pose_mask is None else pose_mask.reshape(-1).astype(bool)
  if inlier_count < 8:
    return None, None, int(inlier_count), inlier_mask

  return rotation, translation.reshape(3), int(inlier_count), inlier_mask


def visual_scale_for_frame(frame_idx, speeds, dt):
  if speeds is None or dt is None or frame_idx <= 0 or frame_idx - 1 >= len(speeds):
    return 1.0
  speed = float(speeds[frame_idx - 1])
  if not np.isfinite(speed):
    return 1.0
  return max(speed, 0.0) * float(dt)


def visual_scale_from_speed(speed_mps, dt):
  if speed_mps is None or dt is None:
    return 1.0
  speed = float(speed_mps)
  if not np.isfinite(speed):
    return 1.0
  return max(speed, 0.0) * float(dt)


def draw_keypoints(frame, keypoints, color=(0, 255, 255), radius=2):
  if keypoints is None:
    return frame
  output = frame.copy()
  for x, y in np.asarray(keypoints, dtype=np.float32):
    cv2.circle(output, (int(round(x)), int(round(y))), radius=radius, color=color, thickness=-1)
  return output


def draw_optical_flow(frame, prev_points, curr_points, line_color=(255, 0, 0), point_color=(0, 255, 255)):
  if prev_points is None or curr_points is None:
    return frame
  prev_points = np.asarray(prev_points, dtype=np.float32)
  curr_points = np.asarray(curr_points, dtype=np.float32)
  if len(prev_points) == 0 or len(curr_points) == 0:
    return frame

  output = frame.copy()
  for prev_point, curr_point in zip(prev_points, curr_points):
    prev_xy = (int(round(prev_point[0])), int(round(prev_point[1])))
    curr_xy = (int(round(curr_point[0])), int(round(curr_point[1])))
    cv2.line(output, prev_xy, curr_xy, color=line_color, thickness=2)
    cv2.circle(output, curr_xy, radius=3, color=point_color, thickness=-1)
  return output


def pose_to_display_transform(pose_row, scale=1.0):
  pose_row = np.asarray(pose_row, dtype=np.float64)
  transform = np.eye(4, dtype=np.float64)
  transform[:3, :3] = euler_to_rotation_matrix(pose_row[3], pose_row[4], pose_row[5])
  transform[:3, 3] = pose_row[:3] * float(scale)
  return transform


class VisualOdometry:
  def __init__(
    self,
    camera_matrix,
    image_scale=0.5,
    max_features=2000,
    ratio=0.75,
    max_hamming_distance=32,
    min_inliers=20,
    ransac_threshold=1.0,
  ):
    self.image_scale = float(image_scale)
    self.camera_matrix = scale_camera_matrix(camera_matrix, self.image_scale)
    self.orb = cv2.ORB_create(nfeatures=int(max_features))
    self.ratio = float(ratio)
    self.max_hamming_distance = int(max_hamming_distance)
    self.min_inliers = int(min_inliers)
    self.ransac_threshold = float(ransac_threshold)

    self.position_cv = np.zeros(3, dtype=np.float64)
    self.rotation_cv = np.eye(3, dtype=np.float64)
    self.prev_keypoints = None
    self.prev_descriptors = None
    self.frame_idx = 0
    self.poses = []
    self.max_features = int(max_features)

  def step(self, frame, speed_mps=None, dt=None):
    processed_frame = frame
    if self.image_scale != 1.0:
      processed_frame = cv2.resize(
        frame,
        None,
        fx=self.image_scale,
        fy=self.image_scale,
        interpolation=cv2.INTER_AREA,
      )
    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    curr_keypoints, curr_descriptors = extract_features(gray, self.orb, self.max_features)

    debug = {
      "keypoints": None,
      "flow_prev": None,
      "flow_curr": None,
      "matches": [],
      "inliers": 0,
      "scale": self.image_scale,
    }

    if self.prev_keypoints is not None:
      matches = match_features(self.prev_descriptors, curr_descriptors, self.ratio, self.max_hamming_distance)
      rel_rotation, rel_translation, inliers, inlier_mask = recover_relative_pose(
        self.prev_keypoints,
        curr_keypoints,
        matches,
        self.camera_matrix,
        self.ransac_threshold,
      )
      debug["matches"] = matches
      debug["inliers"] = int(inliers)

      flow_matches = matches
      if inlier_mask is not None and len(inlier_mask) == len(matches):
        flow_matches = [match for match, keep in zip(matches, inlier_mask) if keep]
      if flow_matches:
        flow_prev = np.asarray([self.prev_keypoints[match.queryIdx] for match in flow_matches], dtype=np.float32)
        flow_curr = np.asarray([curr_keypoints[match.trainIdx] for match in flow_matches], dtype=np.float32)
        if self.image_scale != 1.0:
          flow_prev = flow_prev / self.image_scale
          flow_curr = flow_curr / self.image_scale
        debug["flow_prev"] = flow_prev
        debug["flow_curr"] = flow_curr

      if rel_rotation is not None and inliers >= self.min_inliers:
        scale = visual_scale_from_speed(speed_mps, dt)
        prev_to_curr = -rel_rotation.T.dot(rel_translation) * scale
        self.position_cv += self.rotation_cv.dot(prev_to_curr)
        self.rotation_cv = self.rotation_cv.dot(rel_rotation.T)

    pose = pose_row_from_cv_pose(self.position_cv, self.rotation_cv)
    self.poses.append(pose)

    if self.image_scale != 1.0:
      curr_keypoints = curr_keypoints / self.image_scale
    debug["keypoints"] = curr_keypoints

    self.prev_keypoints = curr_keypoints if self.image_scale == 1.0 else curr_keypoints * self.image_scale
    self.prev_descriptors = curr_descriptors
    self.frame_idx += 1
    return pose, debug


def estimate_visual_odometry(
  video_path,
  camera_matrix,
  speeds=None,
  dt=None,
  max_frames=None,
  image_scale=0.5,
  max_features=2000,
  ratio=0.75,
  max_hamming_distance=32,
  min_inliers=20,
  ransac_threshold=1.0,
  progress=False,
):
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    raise RuntimeError(f"Could not open video for visual odometry: {video_path}")

  vo = VisualOdometry(
    camera_matrix,
    image_scale=image_scale,
    max_features=max_features,
    ratio=ratio,
    max_hamming_distance=max_hamming_distance,
    min_inliers=min_inliers,
    ransac_threshold=ransac_threshold,
  )

  frame_idx = 0
  try:
    while max_frames is None or frame_idx < max_frames:
      ret, frame = cap.read()
      if not ret:
        break

      speed_mps = None if speeds is None or frame_idx <= 0 else speeds[frame_idx - 1]
      pose, _ = vo.step(frame, speed_mps=speed_mps, dt=dt)

      if progress and frame_idx > 0 and frame_idx % 100 == 0:
        print(f"[*] Visual odometry frame {frame_idx}")
      frame_idx += 1
  finally:
    cap.release()

  return np.asarray(vo.poses, dtype=np.float32)
