import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

def to_frame_path(path):
  theta = (3 * np.pi) / 2
  theta = calc_theta(path[1])
  print(theta)

  A = np.matrix([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])

  new_path = np.zeros(path.shape)
  for i,v in enumerate(path):
    new_path[i] = A @ v

  return new_path



if __name__ == '__main__':
  curve1 = np.array([[0,0], [-1,0], [-2,-1], [-2,-2], [-2,-3]])
  curve2 = to_frame_path(curve1)

  plt.scatter(curve1[:, 0], curve1[:, 1])
  plt.plot(curve1[:, 0], curve1[:, 1])
  plt.scatter(curve2[:, 0], curve2[:, 1])
  plt.plot(curve2[:, 0], curve2[:, 1])
  plt.show()
