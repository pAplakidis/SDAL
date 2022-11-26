#!/usr/bin/env python3
import random
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

IMG_WIDTH = 1164
IMG_HEIGHT = 874

actor_list = []

def render_img(img):
  cv2.imshow("frame", img)
  if cv2.waitKey(1) & 0xFF == 27: pass


class Car:
  def __init__(self):
    # TODO: properly init car components
    #self.front_camera = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
    self.front_camera = None
    self.pose = None

  def process_img(self, img):
    img = np.array(img.raw_data)
    img = img.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
    img = img[:, :, :3]
    self.front_camera = img

  def process_imu(self):
    pass

  def process_gps(self):
    pass


# TODO: get camera, GPS, IMU data
def carla_main():
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

  camera.listen(lambda img: car.process_img(img))

  while True:
    if car.front_camera is not None:
      render_img(car.front_camera)


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
    print('done')

