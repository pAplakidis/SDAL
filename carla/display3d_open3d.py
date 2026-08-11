import time
from multiprocessing import Process, Queue

import numpy as np


def _require_open3d():
  try:
    import open3d as o3d
  except ImportError as e:
    raise RuntimeError("open3d is required for 3D replay: python -m pip install open3d") from e
  return o3d


DEFAULT_FRUSTUM_COLOR = (0.0, 1.0, 0.0)
DEFAULT_PATH_COLOR = (1.0, 0.0, 0.0)


def normalize_color(color):
  color = np.asarray(color, dtype=np.float64).reshape(3)
  if np.max(color) > 1.0:
    color = color / 255.0
  return np.clip(color, 0.0, 1.0)


def create_camera_frustum(o3d, scale=0.5, color=DEFAULT_FRUSTUM_COLOR):
  color = normalize_color(color)
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
    self.q = Queue()
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

    def make_stream_state(color):
      return {
        "color": normalize_color(color),
        "poses": None,
        "path": None,
        "frame_idx": 0,
        "frustums": [],
        "drawn_frustums": 0,
        "path_lines": o3d.geometry.LineSet(),
        "current_point": o3d.geometry.PointCloud(),
      }

    def set_lineset_color(lineset, color, line_count):
      if line_count <= 0:
        lineset.colors = o3d.utility.Vector3dVector()
      else:
        lineset.colors = o3d.utility.Vector3dVector([color] * line_count)

    streams = {}
    try:
      while True:
        while not q.empty():
          stream_id, poses, path, frame_idx, color = q.get()
          if stream_id not in streams:
            streams[stream_id] = make_stream_state(color)
            vis.add_geometry(streams[stream_id]["path_lines"])
            vis.add_geometry(streams[stream_id]["current_point"])

          stream = streams[stream_id]
          stream["color"] = normalize_color(color)
          stream["poses"] = poses
          stream["path"] = path
          stream["frame_idx"] = int(frame_idx)

        for stream_id, stream in streams.items():
          poses = stream["poses"]
          path = stream["path"]
          if poses is None:
            continue

          frame_idx = min(stream["frame_idx"], len(poses) - 1, max_frames - 1)

          while len(stream["frustums"]) < min(len(poses), max_frames):
            frustum = create_camera_frustum(o3d, color=stream["color"])
            vis.add_geometry(frustum)
            stream["frustums"].append(frustum)

          if stream["drawn_frustums"] > len(poses):
            # Reset if caller sends shorter pose list for same stream.
            for frustum in stream["frustums"]:
              vis.remove_geometry(frustum, reset_bounding_box=False)
            stream["frustums"] = []
            stream["drawn_frustums"] = 0
            while len(stream["frustums"]) < min(len(poses), max_frames):
              frustum = create_camera_frustum(o3d, color=stream["color"])
              vis.add_geometry(frustum)
              stream["frustums"].append(frustum)

          while (
            stream["drawn_frustums"] <= frame_idx
            and stream["drawn_frustums"] < len(poses)
            and stream["drawn_frustums"] < max_frames
          ):
            frustum = stream["frustums"][stream["drawn_frustums"]]
            frustum.colors = o3d.utility.Vector3dVector([
              stream["color"]
            ] * len(frustum.lines))
            frustum.transform(poses[stream["drawn_frustums"]])
            vis.update_geometry(frustum)
            stream["drawn_frustums"] += 1

          if path is not None and len(path) > 0:
            stream["path_lines"].points = o3d.utility.Vector3dVector(path)
            if len(path) > 1:
              lines = np.column_stack((np.arange(len(path) - 1), np.arange(1, len(path))))
            else:
              lines = np.zeros((0, 2), dtype=np.int32)
            stream["path_lines"].lines = o3d.utility.Vector2iVector(lines)
            set_lineset_color(stream["path_lines"], DEFAULT_PATH_COLOR, len(lines))
            vis.update_geometry(stream["path_lines"])

            curr_idx = min(frame_idx, len(path) - 1)
            stream["current_point"].points = o3d.utility.Vector3dVector(path[curr_idx:curr_idx + 1])
            stream["current_point"].colors = o3d.utility.Vector3dVector([DEFAULT_PATH_COLOR])
            vis.update_geometry(stream["current_point"])

        if not vis.poll_events():
          break
        vis.update_renderer()
        time.sleep(0.01)
    finally:
      vis.destroy_window()

  def draw(self, poses, path=None, frame_idx=0, color=DEFAULT_FRUSTUM_COLOR, stream_id="default"):
    if self.q is None:
      return
    path_copy = None if path is None else path.copy()
    self.q.put((str(stream_id), poses.copy(), path_copy, int(frame_idx), normalize_color(color)))

  def close(self):
    if self.process is not None and self.process.is_alive():
      self.process.terminate()
      self.process.join(timeout=1.0)
    self.process = None
    self.q = None
