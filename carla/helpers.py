import math
import cv2
import numpy as np

"""
try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'linux-x86_64'))[0])
except IndexError as e:
  print("index error", e)
"""
import carla


DESIRE = {
  0: "forward",
  1: "right",
  2: "left"
}

# in carla.LightState enum, the 4th and 5th bit represent the blinkers (on/off)
RIGHT_BLINKER_POS = 4
LEFT_BLINKER_POS = 5

"""
Town01  A small, simple town with a river and several bridges.
Town02	A small simple town with a mixture of residential and commercial buildings.
Town03	A larger, urban map with a roundabout and large junctions.
Town04	A small town embedded in the mountains with a special "figure of 8" infinite highway.
Town05	Squared-grid town with cross junctions and a bridge. It has multiple lanes per direction. Useful to perform lane changes.
Town06	Long many lane highways with many highway entrances and exits. It also has a Michigan left.
Town07	A rural environment with narrow roads, corn, barns and hardly any traffic lights.

NOTE: maps >7 not found
Town08	Secret "unseen" town used for the Leaderboard challenge
Town09	Secret "unseen" town used for the Leaderboard challenge
Town10	A downtown urban environment with skyscrapers, residential buildings and an ocean promenade.
Town11	A Large Map that is undecorated. Serves as a proof of concept for the Large Maps feature.
Town12	A Large Map with numerous different regions, including high-rise, residential and rural environments.
"""
MAPS = [
  "Town01",
  "Town02",
  "Town03",
  "Town04",
  "Town05",
  "Town06",
  "Town07"
]

WEATHERS = {
  "ClearNoon": carla.WeatherParameters.ClearNoon,
  "CloudyNoon": carla.WeatherParameters.CloudyNoon,
  "WetNoon": carla.WeatherParameters.WetNoon,
  "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
  "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
  "HardRainNoon": carla.WeatherParameters.HardRainNoon,
  "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
  "ClearSunset": carla.WeatherParameters.ClearSunset,
  "CloudySunset": carla.WeatherParameters.CloudySunset,
  "WetSunset": carla.WeatherParameters.WetSunset,
  "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
  "MidRainSunset": carla.WeatherParameters.MidRainSunset,
  "HardRainSunset": carla.WeatherParameters.HardRainSunset,
  "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
  "ClearNight": carla.WeatherParameters(cloudiness=0.0,
                                   precipitation=0.0,
                                   sun_altitude_angle=-20.0),
  "CloudyNight": carla.WeatherParameters(cloudiness=80.0,
                                   precipitation=0.0,
                                   sun_altitude_angle=-20.0),
  "HardRainNight": carla.WeatherParameters(cloudiness=80.0,
                                   precipitation=50.0,
                                   sun_altitude_angle=-20.0),
  "SoftRainNight": carla.WeatherParameters(cloudiness=80.0,
                                   precipitation=25.0,
                                   sun_altitude_angle=-20.0)
}


def carla_image_to_bgr(image):
  array = np.frombuffer(image.raw_data, dtype=np.uint8)
  array = array.reshape((image.height, image.width, 4))
  return array[:, :, :3].copy() # CARLA RGB camera actually gives BGRA byte order.

def render_img(img):
  cv2.imshow("CARLA Collector", img)
  if cv2.waitKey(1) & 0xFF == 27:
    raise KeyboardInterrupt


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


DISPLAY_3D_W = 960
DISPLAY_3D_H = 540
TARGET_PATH_EXTENT = 50.0
DEFAULT_LOOKAHEAD = 200
CAMERA_FOV_DEG = 70.0
CAMERA_OFFSET_CARLA = np.array([1.8, 0.0, 1.7], dtype=np.float64)


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

# CARLA semantic segmentation label colors, BGR for OpenCV display.
CITYSCAPES_PALETTE_BGR = np.array([
  [0, 0, 0],        # 0 None
  [70, 70, 70],     # 1 Building
  [40, 40, 100],    # 2 Fence
  [80, 90, 55],     # 3 Other
  [60, 20, 220],    # 4 Pedestrian
  [153, 153, 153],  # 5 Pole
  [50, 234, 157],   # 6 RoadLine
  [128, 64, 128],   # 7 Road
  [232, 35, 244],   # 8 SideWalk
  [35, 142, 107],   # 9 Vegetation
  [142, 0, 0],      # 10 Vehicles
  [156, 102, 102],  # 11 Wall
  [0, 220, 220],    # 12 TrafficSign
  [180, 130, 70],   # 13 Sky
  [81, 0, 81],      # 14 Ground
  [100, 100, 150],  # 15 Bridge
  [140, 150, 230],  # 16 RailTrack
  [180, 165, 180],  # 17 GuardRail
  [30, 170, 250],   # 18 TrafficLight
  [160, 190, 110],  # 19 Static
  [50, 120, 170],   # 20 Dynamic
  [150, 60, 45],    # 21 Water
  [100, 170, 145],  # 22 Terrain
  [255, 255, 255],  # 23 Any
  [142, 0, 0],      # 24 Car
  [70, 0, 0],       # 25 Truck
  [100, 60, 0],     # 26 Bus
  [90, 0, 0],       # 27 Train
  [110, 0, 0],      # 28 Motorcycle
  [100, 80, 0],     # 29 Bicycle
], dtype=np.uint8)


def colorize_segmentation(labels):
  labels = np.asarray(labels, dtype=np.uint8)
  output = np.zeros(labels.shape + (3,), dtype=np.uint8)
  known = labels < len(CITYSCAPES_PALETTE_BGR)
  output[known] = CITYSCAPES_PALETTE_BGR[labels[known]]
  if np.any(~known):
    unknown = labels[~known]
    output[~known] = np.stack(
      ((unknown * 97) % 255, (unknown * 17) % 255, (unknown * 37) % 255),
      axis=1,
    )
  return output


def camera_intrinsics(width, height, fov_deg):
  focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
  return focal, focal, width / 2.0, height / 2.0

def frame_count(cap, arrays):
  counts = [len(value) for value in arrays.values() if isinstance(value, (list, tuple, np.ndarray))]
  video_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if video_count > 0:
    counts.append(video_count)
  return min(counts)

