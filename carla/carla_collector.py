#!/usr/bin/env python3
import random
import math
import glob
import sys
import os
import time
import cv2
import numpy as np
from multiprocessing import Process, Queue
from threading import Thread

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

PATH = []
IMG_WIDTH = 1164
IMG_HEIGHT = 874

out_path = "../collected_data/2/"   # CHANGE THIS
plog_path = "../collected_data/2/"  # CHANGE THIS

actor_list = []

def render_img(img):
  cv2.imshow("DISPLAY", img)
  if cv2.waitKey(1) & 0xFF == 27: pass


class Car:
  def __init__(self):
    # TODO: properly init car components
    #self.front_camera = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
    self.front_camera = None
    self.pose = None

  def process_img(self, img, out_cap):
    img = np.array(img.raw_data)
    img = img.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
    img = img[:, :, :3]
    out_cap.write(img)
    self.front_camera = img

  def process_imu(self, imu):
    self.bearing_deg = math.degrees(imu.compass)
    self.acceleration = [imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z]
    self.gyro = [imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]

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


# TODO: using carla's locations instead of GNSS, visual odometry, etc is just a temp fix
def carla_main():
  # TODO: out video needs to be hevc format
  #fourcc = cv2.CV_FOURCC(*'MP4V')
  out = cv2.VideoWriter(out_path+"video.mp4", cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (IMG_WIDTH, IMG_HEIGHT))

  # setup
  client = carla.Client('localhost', 2000)
  client.set_timeout(2.0)  # seconds
  world = client.get_world()
  world.set_weather(carla.WeatherParameters.ClearSunset)
  #print(world.get_weather)
  bp_lib = world.get_blueprint_library()

  car = Car()

  # spawn a car
  vehicle_bp = bp_lib.filter('vehicle.tesla.*')[1]
  spawn_point = random.choice(world.get_map().get_spawn_points())
  vehicle = world.spawn_actor(vehicle_bp, spawn_point)

  # make tires less slippery
  # wheel_control = carla.WheelPhysicsControl(tire_friction=5)
  physics_control = vehicle.get_physics_control()
  physics_control.mass = 2326
  # physics_control.wheels = [wheel_control]*4
  physics_control.torque_curve = [[20.0, 500.0], [5000.0, 500.0]]
  physics_control.gear_switch_time = 0.0
  vehicle.apply_physics_control(physics_control)

  # temp controls
  #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
  vehicle.set_autopilot(True)

  actor_list.append(vehicle)

  # spawn camera
  camera_bp = bp_lib.find('sensor.camera.rgb')
  camera_bp.set_attribute('image_size_x', f'{IMG_WIDTH}')
  camera_bp.set_attribute('image_size_y', f'{IMG_HEIGHT}')
  camera_bp.set_attribute('fov', '70')
  camera_bp.set_attribute('sensor_tick', '0.05')
  spawn_point  = carla.Transform(carla.Location(x=0.8, z=1.13))
  camera = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)
  actor_list.append(camera_bp)
  camera.listen(lambda img: car.process_img(img, out))

  # spawn IMU
  imu_bp = bp_lib.find("sensor.other.imu")
  imu = world.spawn_actor(imu_bp, spawn_point, attach_to=vehicle)
  imu.listen(lambda imu: car.process_imu(imu))

  # spawn GPS
  gps_bp = bp_lib.find("sensor.other.gnss")
  gps = world.spawn_actor(gps_bp, spawn_point, attach_to=vehicle)
  gps.listen(lambda gps: car.process_gps(gps))

  # mainloop
  # TODO: car moves like crazy after a while!!!
  frame_id = 0  # TODO: frames from different sensors are not synced
  try:
    while True:
      lx,ly,lz = vehicle.get_location().x, vehicle.get_location().y ,vehicle.get_location().z
      #rx,ry,rz = vehicle.get_orientation().x, vehicle.get_orientation().y ,vehicle.get_orientation().z

      if car.front_camera is not None:
        render_img(car.front_camera)
        print("[+] Frame: ", frame_id, "=>", car.front_camera.shape)
        print("[+] Car Location: (x y z)=(", lx,ly,lz, ")")
        PATH.append((lx, ly, lz))
        #print("[+] Car Rotation: (x y z)=(", rx,ry,rz, ")")
        print("[->] IMU DATA => acceleration", car.acceleration, " : gyroscope", car.gyro)
        print("[->] GNSS DATA => latitude", car.gps_location['latitude'],
              " : longtitude", car.gps_location['longitude'],
              " : altitude", car.gps_location['altitude'])
        print()
        frame_id += 1
  except KeyboardInterrupt:
    print("[~] Stopped recording")
  
  out.release()
  print("[+] Camera recordings saved at: ", out_path+"video.mp4")
  path = np.array(PATH)
  print(path)
  print(path.shape)

  # TODO: preprocess path so that it's coordinates can be projected on 2D display
  #np.save(plog_path, np.array(PATH))


if __name__ == '__main__':
  print("Hello")
  try:
    carla_main()
  except RuntimeError:
    print("Restarting ...")
  finally:
    print("destroying all actors")
    for a in actor_list:
      a.destroy()
    cv2.destroyAllWindows()
    print('Done')
