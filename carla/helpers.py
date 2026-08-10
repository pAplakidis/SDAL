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
