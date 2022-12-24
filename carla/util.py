import numpy as np

IMG_WIDTH = 1164
IMG_HEIGHT = 874
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


# TODO: display path on 3D as well (use pangolin)
def get_local_path(path):
  start_pos = path[0]
  local_path = []
  for i in range(len(path)):
    local_path.append(path[i] - start_pos)
  return np.array(local_path)

def img_from_device(device_path):
  in_shape = device_path.shape
  device_path = np.atleast_2d(device_path)
  path_view = np.einsum('jk, ik->ij', view_frame_from_device_frame, device_path)

  #path_view[path_view[:, 2] < 0] = np.nan  # TODO: commenting this out is a temp hack
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