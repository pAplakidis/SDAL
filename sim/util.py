import cv2
import math
import numpy as np
import io
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

IMG_WIDTH = 1164  # 2D camera display W
IMG_HEIGHT = 874  # 2D camera display W
RW = 1920//2  # 3D renderer W
RH = 1080//2  # 3D renderer H
FPATH_2D_W = IMG_WIDTH//2   # display H for live 2D frame path
FPATH_2D_H = IMG_HEIGHT//2  # display H for live 2D frame path
REC_TIME = 60 # recording length in seconds
LOOKAHEAD = 200

FRAME_TIME = 50

FOCAL = 910.0

# device/mesh : x->forward, y-> right, z->down
# view : x->right, y->down, z->forward
device_frame_from_view_frame = np.array([
  [ 0.,  0.,  1.],
  [ 1.,  0.,  0.],
  [ 0.,  1.,  0.]
])
view_frame_from_device_frame = device_frame_from_view_frame.T

# aka 'K' aka camera_frame_from_view_frame
cam_intrinsics = np.array([
  [FOCAL,   0.,   IMG_WIDTH/2.],
  [  0.,  FOCAL,  IMG_HEIGHT/2.],
  [  0.,    0.,     1.]])

# aka 'K_inv' aka view_frame_from_camera_frame
eon_intrinsics_inv = np.linalg.inv(cam_intrinsics)


"""
def quat2rot(quats):
  input_shape = quats.shape
  quats = np.atleast_2d(quats)
  Rs = np.zeros((quats.shape[0], 3, 3))
  q0 = quats[:, 0]
  q1 = quats[:, 1]
  q2 = quats[:, 2]
  q3 = quats[:, 3]
  Rs[:, 0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
  Rs[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
  Rs[:, 0, 2] = 2 * (q0 * q2 + q1 * q3)
  Rs[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
  Rs[:, 1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
  Rs[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
  Rs[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
  Rs[:, 2, 1] = 2 * (q0 * q1 + q2 * q3)
  Rs[:, 2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

  if len(input_shape) < 2:
    return Rs[0]
  else:
    return Rs
"""

# TODO: needs more work, the car faces the wrong way
# orientation (quaterium) to rotation
def to_rotation_matrix(rotation):
  # rotation = [pitch roll yaw] meaning [y z x]
  # [a b c] = [z y x]
  a, b, c = rotation
  #a, b, c = rotation[2], rotation[0], rotation[1]
  r = [[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]]

  r[0][0] = math.cos(a) * math.cos(b)
  r[0][1] = math.cos(a)*math.sin(b)*math.sin(c) - math.sin(a)*math.cos(c)
  r[0][2] = math.cos(a)*math.sin(b)*math.cos(c) + math.sin(a)*math.sin(c)
  r[1][0] = math.sin(a) * math.cos(b)
  r[1][1] = math.sin(a)*math.sin(b)*math.sin(c) + math.cos(a)*math.cos(c)
  r[1][2] = math.sin(a)*math.sin(b)*math.cos(c) - math.cos(a)*math.sin(c)
  r[2][0] = - math.sin(b)
  r[2][1] = math.cos(b) * math.sin(c)
  r[2][2] = math.cos(b) * math.cos(c)

  return np.array(r)

def to_Rt(t, R):
  Rt = np.eye(4)
  Rt[:3, :3] = R
  Rt[:3, 3] = t
  return Rt

# used on 3D display for converting global carla path to a local one, starting from <0,0,0>
# this function just starts everything from pos[0][0], while it needs to normalize everything relative to the car instead of global positions
# NOTE: since the data comes from UE4, (x,y,z) needs to be converted to (x,z,y)
# poses = [[location(x,y,z), forward_vector(x,y,z)], [..., ...], ...]
def get_relative_poses(poses):
  #print(poses.shape)
  path = poses[:, 0]    # [x y z]
  fvector = poses[:, 1] # [pitch roll yaw] meaning [y z x]

  #path[:, [1, 2]] = path[:, [2, 1]]  # swap y and z because UE4

  start_pos = path[0]
  start_rot = fvector[0]

  local_poses = []
  local_path = []
  local_orients = []
  for i in range(len(path)):
    path[i][0] = -path[i][0]
    local_pos = path[i] - start_pos
    #local_rot = fvector[i] - start_rot  # NOTE: we shouldn't modify rotations
    local_rot = fvector[i]
    local_orient = to_rotation_matrix(local_rot)
    local_orients.append(local_orient)
    local_path.append(local_pos)
    local_poses.append(to_Rt(local_pos, local_orient))

  return np.array(local_poses), np.array(local_path), np.array(local_orients)

def get_relative_path(path):
  start_pos = path[0]
  rel_path = []
  for i in range(len(path)):
    rel_path.append(path[i] - start_pos)
  return np.array(rel_path)

# ------------------------------------------------------------
# returns the unit vector of a vector
def unit_vector(vector):
  return vector / np.linalg.norm(vector)

# returns the angle between vectors v1 and v2 in radians
def angle_between(v1, v2):
  v1_u = unit_vector(v1)
  v2_u = unit_vector(v2)
  return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# returns angle of a vector and positive y-axis clockwise
def calc_theta(vector):
  y_pos = [0,1]
  angle = angle_between(vector, y_pos)
  angle_deg = 360*angle/(2*np.pi)
  if vector[0] < 0:
    angle_deg = 360 - angle_deg
  return math.radians(angle_deg)

# TODO: maybe do this for all frames once and for all instead of redoing it for each frame (more efficient + cleaner dataset/code)
def get_frame_path(path):
  relative_path = get_relative_path(path)
  path_2d = relative_path[:, :2]
  forward_vector = [path_2d[1][0] - path_2d[0][0], path_2d[1][1] - path_2d[0][1]]
  theta = calc_theta(forward_vector)

  A = np.matrix([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])

  new_path = np.zeros(path_2d.shape)
  for i,v in enumerate(path_2d):
    new_path[i] = A @ v

  return path, new_path
# ------------------------------------------------------------

def img_from_device(device_path):
  in_shape = device_path.shape
  device_path = np.atleast_2d(device_path)
  path_view = np.einsum('jk, ik->ij', view_frame_from_device_frame, device_path)

  path_view[path_view[:, 2] < 0] = np.nan  # TODO: commenting this out is a temp hack
  img_path = path_view/path_view[:,2:3]
  return img_path.reshape(in_shape)[:,:2]

def normalize(img_pts):
  # normalizes image coordinates
  # accepts single pt or array of pts
  img_pts = np.array(img_pts)
  input_shape = img_pts.shape
  img_pts = np.atleast_2d(img_pts)
  img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0],1))))
  img_pts_normalized = eon_intrinsics_inv.dot(img_pts.T).T
  img_pts_normalized[(img_pts < 0).any(axis=1)] = np.nan
  return img_pts_normalized[:,:2].reshape(input_shape)

def denormalize(img_pts):
  # denormalizes image coordinates
  # accepts single pt or array of pts
  img_pts = np.array(img_pts)
  input_shape = img_pts.shape
  img_pts = np.atleast_2d(img_pts)
  img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0],1))))
  img_pts_denormalized = cam_intrinsics.dot(img_pts.T).T
  img_pts_denormalized[img_pts_denormalized[:,0] > IMG_WIDTH] = np.nan
  img_pts_denormalized[img_pts_denormalized[:,0] < 0] = np.nan
  img_pts_denormalized[img_pts_denormalized[:,1] > IMG_HEIGHT] = np.nan
  img_pts_denormalized[img_pts_denormalized[:,1] < 0] = np.nan
  return img_pts_denormalized[:,:2].reshape(input_shape)

def draw_path(path, img, width=1, height=1.2, fill_color=(128,0,255), line_color=(0,0, 255)):
  # TODO: debug 3D to 2D convertion
  img_points_norm = img_from_device(path) # TODO: this outputs NAN
  print("<2D-RENDERER> normalized img pts:", len(img_points_norm))
  img_pts = denormalize(img_points_norm)
  print("<2D-RENDERER> denormalized img pts:", len(img_pts))
  valid = np.isfinite(img_pts).all(axis=1)
  img_pts = img_pts[valid].astype(int)


  print("<2D-RENDERER> valid img pts:", len(img_pts))
  for i in range(1, len(img_pts)):
    #print(img_pts[i])
    cv2.circle(img, img_pts[i], 1, line_color, -1)

def figshow(fig):
  buf = io.BytesIO()
  pio.write_image(fig, buf)
  buf.seek(0)
  file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  cv2.imshow("Path Plot", img)

DESIRE = {0: "forward",
          1: "right",
          2: "left"}

# specifically for desire (0, 1, 2)
def one_hot_encode(arr):
  new_arr = []
  for i in range(len(arr)):
    idx = arr[i]
    tmp = [0, 0, 0]
    tmp[idx] = 1
    new_arr.append(tmp)
  return np.array(new_arr)
