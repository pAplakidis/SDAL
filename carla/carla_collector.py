import os
import random
from tqdm import trange
from queue import Queue, Empty

import carla
import cv2
import numpy as np

from carla_config import *
from helpers import *
from hevc_writer import HEVCWriter


OUT_PATH = os.getenv("OUT_PATH")
if OUT_PATH is None:
  print("Usage: OUT_PATH=<output_path> MAP=<map_idx> python carla_collector.py")
  raise SystemExit(1)
os.makedirs(OUT_PATH, exist_ok=True)

RENDER = int(os.getenv("RENDER", 0))

MAP = int(os.getenv("MAP", 0))
curr_map = MAPS[MAP]

WEATHER_KEY = os.getenv("WEATHER", "CloudyNoon")
WEATHER = WEATHERS[WEATHER_KEY]

VIDEO_ENCODER = os.getenv("VIDEO_ENCODER", "libx265")
VIDEO_CRF = int(os.getenv("VIDEO_CRF", 23))
VIDEO_PRESET = os.getenv("VIDEO_PRESET", "medium")


PLOG_POSES = os.path.join(OUT_PATH, "poses.npy")
PLOG_DESIRES = os.path.join(OUT_PATH, "desires.npy")
PLOG_STEERING = os.path.join(OUT_PATH, "steering_angles.npy")
PLOG_THROTTLE = os.path.join(OUT_PATH, "throttles.npy")
PLOG_SPEEDS = os.path.join(OUT_PATH, "speeds.npy")
PLOG_IMU = os.path.join(OUT_PATH, "imu.npy")
PLOG_GNSS = os.path.join(OUT_PATH, "gnss.npy")

VIDEO_PATH = os.path.join(OUT_PATH, "video.hevc")
SEGMENTATION_PATH = os.path.join(OUT_PATH, "segmentation.npy")


actor_list = []
vehicles = []
walkers = []
walker_controllers = []


def get_sensor_for_frame(queue, target_frame, timeout=5.0):
  while True:
    # print(f"waiting target={target_frame}, qsize={queue.qsize()}")

    try:
      data = queue.get(timeout=timeout)
    except Empty:
      raise RuntimeError(f"Timed out waiting for sensor frame {target_frame}")

    # print(f"target={target_frame}, received={data.frame}, timestamp={data.timestamp}")

    if data.frame == target_frame:
      return data

    if data.frame > target_frame:
      raise RuntimeError(f"Sensor jumped past requested frame {target_frame} -> {data.frame}")


def carla_segmentation_to_labels(image):
  array = np.frombuffer(image.raw_data, dtype=np.uint8)
  array = array.reshape((image.height, image.width, 4))
  return array[:, :, 2].copy()

def spawn_traffic(world, traffic_manager):
  bp_lib = world.get_blueprint_library()
  spawn_points = world.get_map().get_spawn_points()

  random.shuffle(spawn_points)

  for spawn_point in spawn_points[:N_VEHICLES]:
    bp = random.choice(bp_lib.filter("vehicle.*"))

    vehicle = world.try_spawn_actor(bp, spawn_point)

    if vehicle is None:
      continue

    vehicle.set_autopilot(
      True,
      traffic_manager.get_port(),
    )

    vehicles.append(vehicle)
    actor_list.append(vehicle)

  print(f"[*] Spawned {len(vehicles)} traffic vehicles")


def spawn_pedestrians(client, world):
  blueprint_library = world.get_blueprint_library()
  walker_blueprints = blueprint_library.filter("walker.pedestrian.*")

  walker_spawn_points = []
  for _ in range(N_PEDESTRIANS):
    location = world.get_random_location_from_navigation()
    if location is None:
      continue
    walker_spawn_points.append(carla.Transform(location))

  # Spawn walkers
  batch = []
  for spawn_point in walker_spawn_points:
    bp = random.choice(walker_blueprints)
    if bp.has_attribute("is_invincible"):
      bp.set_attribute("is_invincible", "false")
    batch.append(carla.command.SpawnActor(bp, spawn_point))

  results = client.apply_batch_sync(batch, False)
  walker_ids = []
  for result in results:
    if result.error:
      print(f"[!] Walker spawn error: {result.error}")
      continue
    walker_ids.append(result.actor_id)

  # Allow newly spawned walkers to become available.
  world.tick()

  # Spawn AI controllers
  controller_bp = blueprint_library.find("controller.ai.walker")
  batch = []
  for walker_id in walker_ids:
    batch.append(carla.command.SpawnActor(controller_bp, carla.Transform(), walker_id))

  controller_results = client.apply_batch_sync(batch, False)
  controller_ids = []
  valid_walker_ids = []
  for walker_id, result in zip(walker_ids, controller_results):
    if result.error:
      print(f"[!] Walker controller spawn error: {result.error}")
      continue
    valid_walker_ids.append(walker_id)
    controller_ids.append(result.actor_id)

  world.tick()

  # Get actual Actor objects
  for walker_id in valid_walker_ids:
    walker = world.get_actor(walker_id)
    if walker is not None:
      walkers.append(walker)
      actor_list.append(walker)

  for controller_id in controller_ids:
    controller = world.get_actor(controller_id)
    if controller is None:
      continue
    walker_controllers.append(controller)
    actor_list.append(controller)

  for controller in walker_controllers:
    controller.start()
    destination = world.get_random_location_from_navigation()
    if destination is not None:
      controller.go_to_location(destination)
    controller.set_max_speed(random.uniform(1.0, 2.0))

  # Allows some pedestrians to cross roads.
  world.set_pedestrians_cross_factor(0.1)
  print(f"[*] Spawned {len(walkers)} pedestrians with {len(walker_controllers)} controllers")


def configure_vehicle_physics(vehicle):
  physics_control = vehicle.get_physics_control()

  physics_control.mass = 2326

  # Do NOT replace the WheelPhysicsControl objects completely.
  # Keep their existing radius, damping, brake values, etc.
  for wheel in physics_control.wheels:
    wheel.tire_friction = 5.0

  physics_control.torque_curve = [
    carla.Vector2D(20.0, 500.0),
    carla.Vector2D(5000.0, 500.0),
  ]
  physics_control.gear_switch_time = 0.0
  vehicle.apply_physics_control(physics_control)


def spawn_ego_vehicle(world, traffic_manager):
  bp_lib = world.get_blueprint_library()
  try:
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
  except RuntimeError:
    vehicle_bp = random.choice(bp_lib.filter("vehicle.tesla.*"))

  spawn_points = world.get_map().get_spawn_points()
  random.shuffle(spawn_points)

  vehicle = None
  for spawn_point in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is not None:
      break

  if vehicle is None:
    raise RuntimeError("Could not spawn ego vehicle")

  configure_vehicle_physics(vehicle)
  vehicle.set_autopilot(True, traffic_manager.get_port())
  traffic_manager.update_vehicle_lights(vehicle, True)
  actor_list.append(vehicle)
  print("[*] Ego vehicle spawned")
  return vehicle


def spawn_sensors(world, vehicle):
  bp_lib = world.get_blueprint_library()

  camera_queue = Queue()
  segmentation_queue = Queue()
  imu_queue = Queue()
  gnss_queue = Queue()

  camera_transform = carla.Transform(
    carla.Location(
      x=1.8,
      z=1.7,
    )
  )

  aux_transform = carla.Transform(
    carla.Location(
      x=0.8,
      z=1.13,
    )
  )

  # Camera
  camera_bp = bp_lib.find("sensor.camera.rgb")
  camera_bp.set_attribute("image_size_x", str(IMG_WIDTH))
  camera_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
  camera_bp.set_attribute("fov", "70")
  # camera_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))
  camera_bp.set_attribute("sensor_tick", "0.0")
  camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
  camera.listen(camera_queue.put)
  actor_list.append(camera)
  print("[*] Camera spawned")

  # Semantic segmentation camera, aligned with RGB camera.
  segmentation_bp = bp_lib.find("sensor.camera.semantic_segmentation")
  segmentation_bp.set_attribute("image_size_x", str(IMG_WIDTH))
  segmentation_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
  segmentation_bp.set_attribute("fov", "70")
  segmentation_bp.set_attribute("sensor_tick", "0.0")
  segmentation_camera = world.spawn_actor(segmentation_bp, camera_transform, attach_to=vehicle)
  segmentation_camera.listen(segmentation_queue.put)
  actor_list.append(segmentation_camera)
  print("[*] Segmentation camera spawned")

  # IMU
  imu_bp = bp_lib.find("sensor.other.imu")
  # imu_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))
  imu_bp.set_attribute("sensor_tick", "0.0.")
  imu = world.spawn_actor(imu_bp, aux_transform, attach_to=vehicle)
  imu.listen(imu_queue.put)
  actor_list.append(imu)
  print("[*] IMU spawned")

  # GNSS
  gnss_bp = bp_lib.find("sensor.other.gnss")
  # gnss_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))
  gnss_bp.set_attribute("sensor_tick", "0.0")
  gnss = world.spawn_actor(gnss_bp, aux_transform, attach_to=vehicle)
  gnss.listen(gnss_queue.put)
  actor_list.append(gnss)
  print("[*] GNSS spawned")

  return (
    camera,
    segmentation_camera,
    imu,
    gnss,
    camera_queue,
    segmentation_queue,
    imu_queue,
    gnss_queue,
  )


def init_carla():
  client = carla.Client("localhost", 2000)
  client.set_timeout(10.0)
  print(f"[+] Client connected")

  print(f"[*] Loading map: {curr_map}")
  world = client.load_world(curr_map)
  world.set_weather(WEATHER)

  original_settings = world.get_settings()

  # Configure synchronous mode BEFORE collecting sensor data.
  settings = world.get_settings()
  settings.synchronous_mode = True
  settings.no_rendering_mode = False
  settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
  world.apply_settings(settings)

  traffic_manager = client.get_trafficmanager()
  traffic_manager.set_synchronous_mode(True)
  traffic_manager.set_global_distance_to_leading_vehicle(2.5)
  traffic_manager.set_respawn_dormant_vehicles(True)

  print("[*] Synchronous simulation enabled")

  spawn_traffic(world, traffic_manager)
  spawn_pedestrians(client, world)
  vehicle = spawn_ego_vehicle(world, traffic_manager)
  sensors = spawn_sensors(world, vehicle)

  return (
    client,
    world,
    traffic_manager,
    vehicle,
    original_settings,
    sensors,
  )


def get_desire(vehicle):
  light_state = vehicle.get_light_state()
  right_blinker = bool(light_state & (1 << RIGHT_BLINKER_POS))
  left_blinker = bool(light_state & (1 << LEFT_BLINKER_POS))

  if right_blinker and not left_blinker:
    return 1
  if left_blinker and not right_blinker:
    return 2
  return 0


def get_speed(vehicle):
  velocity = vehicle.get_velocity()
  # m/s
  return np.sqrt(
    velocity.x ** 2 +
    velocity.y ** 2 +
    velocity.z ** 2
  )


def mainloop(world, vehicle, traffic_manager, camera_queue, segmentation_queue, imu_queue, gnss_queue, video_writer):
  poses = []
  desires = []
  steering_angles = []
  throttles = []
  speeds = []
  imu_data = []
  gnss_data = []

  total_frames = int(REC_TIME / FIXED_DELTA_SECONDS)
  segmentation_data = np.lib.format.open_memmap(
    SEGMENTATION_PATH,
    mode="w+",
    dtype=np.uint8,
    shape=(total_frames, IMG_HEIGHT, IMG_WIDTH),
  )
  print(f"[*] Recording {REC_TIME}s = {total_frames} frames at {FPS} FPS")
  for frame_index in (t := trange(total_frames)):
    # Advance the entire simulation exactly one fixed step.
    carla_frame = world.tick()

    # Get data generated by THAT simulation frame.
    image = get_sensor_for_frame(camera_queue, carla_frame)
    segmentation_image = get_sensor_for_frame(segmentation_queue, carla_frame)
    imu = get_sensor_for_frame(imu_queue, carla_frame)
    gnss = get_sensor_for_frame(gnss_queue, carla_frame)
    frame = carla_image_to_bgr(image)
    segmentation_data[frame_index] = carla_segmentation_to_labels(segmentation_image)

    # Video
    video_writer.write(frame)
    if RENDER:
      render_img(frame)

    # Vehicle state
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    control = vehicle.get_control()
    desire = get_desire(vehicle)
    speed = get_speed(vehicle)
    poses.append([
      location.x,
      location.y,
      location.z,
      rotation.roll,
      rotation.pitch,
      rotation.yaw,
    ])

    desires.append(desire)
    steering_angles.append(control.steer)
    throttles.append(control.throttle)
    speeds.append(speed)

    # IMU
    imu_data.append([
      imu.accelerometer.x,
      imu.accelerometer.y,
      imu.accelerometer.z,
      imu.gyroscope.x,
      imu.gyroscope.y,
      imu.gyroscope.z,
      imu.compass,
    ])

    # GNSS
    gnss_data.append([
      gnss.latitude,
      gnss.longitude,
      gnss.altitude,
    ])

    if frame_index % FPS == 0:
      t.set_description(
        f"CARLA frame {carla_frame} "
        f"| speed={speed:.2f} m/s "
        f"| steer={control.steer:.3f} "
        f"| throttle={control.throttle:.3f} "
        f"| desire={DESIRE[desire]}"
      )

  segmentation_data.flush()

  return {
    "poses": np.asarray(poses, dtype=np.float32),
    "desires": np.asarray(desires, dtype=np.uint8),
    "steering_angles": np.asarray(steering_angles, dtype=np.float32),
    "throttles": np.asarray(throttles, dtype=np.float32),
    "speeds": np.asarray(speeds, dtype=np.float32),
    "imu": np.asarray(imu_data, dtype=np.float32),
    "gnss": np.asarray(gnss_data, dtype=np.float64),
  }


def save_data(data):
  np.save(PLOG_POSES, data["poses"])
  np.save(PLOG_DESIRES, data["desires"])
  np.save(PLOG_STEERING, data["steering_angles"])
  np.save(PLOG_THROTTLE, data["throttles"])
  np.save(PLOG_SPEEDS, data["speeds"])
  np.save(PLOG_IMU, data["imu"])
  np.save(PLOG_GNSS, data["gnss"])
  print(f"[+] Video:             {VIDEO_PATH}")
  print(f"[+] Segmentation:      {SEGMENTATION_PATH}")
  print(f"[+] Poses:             {PLOG_POSES}")
  print(f"[+] Desires:           {PLOG_DESIRES}")
  print(f"[+] Steering angles:   {PLOG_STEERING}")
  print(f"[+] Throttles:         {PLOG_THROTTLE}")
  print(f"[+] Speeds:            {PLOG_SPEEDS}")
  print(f"[+] IMU:               {PLOG_IMU}")
  print(f"[+] GNSS:              {PLOG_GNSS}")


def cleanup(world, traffic_manager, original_settings):
  print("[*] Cleaning up CARLA actors")

  # Stop walker AI before destroying controllers.
  for controller in walker_controllers:
    try:
      if controller.is_alive:
        controller.stop()
    except RuntimeError:
      pass

  # Stop sensors before destruction.
  for actor in actor_list:
    try:
      if actor.is_alive and actor.type_id.startswith("sensor."):
        actor.stop()
    except RuntimeError:
      pass

  for actor in reversed(actor_list):
    try:
      if actor.is_alive:
        actor.destroy()
    except RuntimeError:
      pass

  try:
    traffic_manager.set_synchronous_mode(False)
  except RuntimeError:
    pass

  try:
    world.apply_settings(original_settings)
  except RuntimeError:
    pass

  if RENDER:
    cv2.destroyAllWindows()

  print("[+] Done")


def carla_main():
  print(f"[*] Weather: {WEATHER_KEY}")
  print(f"[*] Output: {OUT_PATH}")

  client = None
  world = None
  traffic_manager = None
  original_settings = None
  video_writer = None

  try:
    (
      client,
      world,
      traffic_manager,
      vehicle,
      original_settings,
      sensors,
    ) = init_carla()

    (
      camera,
      segmentation_camera,
      imu,
      gnss,
      camera_queue,
      segmentation_queue,
      imu_queue,
      gnss_queue,
    ) = sensors

    # One initialization frame lets all attached sensors become active.
    # world.tick()
    init_frame = world.tick()
    print(f"[*] Initializing sensors at frame {init_frame}")
    get_sensor_for_frame(camera_queue, init_frame)
    get_sensor_for_frame(segmentation_queue, init_frame)
    get_sensor_for_frame(imu_queue, init_frame)
    get_sensor_for_frame(gnss_queue, init_frame)
    print("[*] Sensors synchronized")

    video_writer = HEVCWriter(
      VIDEO_PATH,
      IMG_WIDTH,
      IMG_HEIGHT,
      FPS,
      encoder=VIDEO_ENCODER,
      crf=VIDEO_CRF,
      preset=VIDEO_PRESET,
    )

    data = mainloop(
      world,
      vehicle,
      traffic_manager,
      camera_queue,
      segmentation_queue,
      imu_queue,
      gnss_queue,
      video_writer,
    )

    video_writer.close()
    video_writer = None

    save_data(data)

  except KeyboardInterrupt:
    print("[~] Recording interrupted")

  finally:
    if video_writer is not None:
      try:
        video_writer.close()
      except Exception as e:
        print(f"[!] Error closing video writer: {e}")

    if (
      world is not None
      and traffic_manager is not None
      and original_settings is not None
    ):
      cleanup(
        world,
        traffic_manager,
        original_settings,
      )


if __name__ == "__main__":
  carla_main()
