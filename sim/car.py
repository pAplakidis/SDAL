import time
import math
import numpy as np
from .carla_config import IMG_HEIGHT, IMG_WIDTH


# TODO: move vehicle logic from mainloop in here
class Car:
  def __init__(self, frames, poses, desires):
    # TODO: properly init car components
    #self.front_camera = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
    self.front_camera = None
    self.pose = None
    self.gyro = None

    self.frames = frames
    self.poses = poses
    self.desires = desires

  def process_img(self, img, location, rotation, desire):
    img = np.array(img.raw_data)
    img = img.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
    img = img[:, :, :3]

    if location is not None and rotation is not None and desire is not None:
      self.frames.append(img)
      self.poses.append((location, rotation))
      self.desires.append(desire)
    self.front_camera = img

  def process_imu(self, imu):
    self.bearing_deg = math.degrees(imu.compass)  # radians
    self.acceleration = [imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z] # m/s**2
    self.gyro = [imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z] # radians

  def process_gps(self, gps):
    # TODO: update this
    self.gps_location = {
      "timestamp": int(time.time() * 1000),
      "accuracy": 1.0,
      "speed_accuracy": 0.1,
      "bearing_accuracy_deg": 0.1,
      "bearing_deg": self.bearing_deg,
      "latitude": gps.latitude,
      "longitude": gps.longitude,
      "altitude": gps.altitude,
      "speed": 0,
    }


