import time
from multiprocessing import Process, Queue

import numpy as np


def _require_open3d():
  try:
    import open3d as o3d
  except ImportError as e:
    raise RuntimeError("open3d is required for 3D replay: python -m pip install open3d") from e
  return o3d


def create_camera_frustum(o3d, scale=0.5, color=(0.0, 1.0, 0.0)):
  pts = np.array([
    [0.0, 0.0, 0.0],
    [-0.5, -0.5, -1.0],
    [0.5, -0.5, -1.0],
    [0.5, 0.5, -1.0],
    [-0.5, 0.5, -1.0],
  ]) * scale
  lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
  frustum = o3d.geometry.LineSet()
  frustum.points = o3d.utility.Vector3dVector(pts)
  frustum.lines = o3d.utility.Vector2iVector(lines)
  frustum.colors = o3d.utility.Vector3dVector([color] * len(lines))
  return frustum


class Display3D:
  def __init__(self, width, height, max_frames=2000, title="Display 3D"):
    _require_open3d()
    self.width = width
    self.height = height
    self.max_frames = max_frames
    self.title = title
    self.q = Queue(maxsize=2)
    self.process = Process(
      target=self._viewer_main,
      args=(self.q, self.width, self.height, self.max_frames, self.title),
      daemon=True,
    )
    self.process.start()

  @staticmethod
  def _viewer_main(q, width, height, max_frames, title):
    o3d = _require_open3d()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=width, height=height)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])
    opt.point_size = 3.0

    ctr = vis.get_view_control()
    ctr.set_constant_z_far(1000.0)
    ctr.set_constant_z_near(0.01)
    ctr.set_zoom(0.5)

    frustums = []
    for _ in range(max_frames):
      frustum = create_camera_frustum(o3d)
      vis.add_geometry(frustum)
      frustums.append(frustum)

    path_lines = o3d.geometry.LineSet()
    vis.add_geometry(path_lines)

    current_point = o3d.geometry.PointCloud()
    vis.add_geometry(current_point)

    state = None
    drawn_frustums = 0
    try:
      while True:
        while not q.empty():
          state = q.get()

        if state is not None:
          poses, path, frame_idx = state
          frame_idx = min(frame_idx, len(poses) - 1, max_frames - 1)

          while drawn_frustums <= frame_idx and drawn_frustums < len(poses) and drawn_frustums < max_frames:
            frustums[drawn_frustums].transform(poses[drawn_frustums])
            vis.update_geometry(frustums[drawn_frustums])
            drawn_frustums += 1

          if len(path) > 0:
            path_lines.points = o3d.utility.Vector3dVector(path)
            if len(path) > 1:
              lines = np.column_stack((np.arange(len(path) - 1), np.arange(1, len(path))))
            else:
              lines = np.zeros((0, 2), dtype=np.int32)
            path_lines.lines = o3d.utility.Vector2iVector(lines)
            path_lines.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]] * len(lines))
            vis.update_geometry(path_lines)

            current_point.points = o3d.utility.Vector3dVector(path[frame_idx:frame_idx + 1])
            current_point.colors = o3d.utility.Vector3dVector([[0.0, 0.4, 1.0]])
            vis.update_geometry(current_point)

        if not vis.poll_events():
          break
        vis.update_renderer()
        time.sleep(0.01)
    finally:
      vis.destroy_window()

  def draw(self, poses, path, frame_idx):
    if self.q is None:
      return
    while self.q.full():
      try:
        self.q.get_nowait()
      except Exception:
        break
    self.q.put((poses.copy(), path.copy(), int(frame_idx)))

  def close(self):
    if self.process is not None and self.process.is_alive():
      self.process.terminate()
      self.process.join(timeout=1.0)
    self.process = None
    self.q = None
