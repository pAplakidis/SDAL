import numpy as np
import pangolin
import OpenGL.GL as gl
from multiprocessing import Process, Queue

from util import *

class Renderer3D:
  def __init__(self, w, h):
    self.W = w
    self.H = h
    self.poses = None
    self.poses_q = Queue()
    self.path_q = Queue()
    self.path = None
    self.poses = None
    self.p = Process(target=self.renderer_main, args=(self.poses_q, self.path_q,))
    #self.p.daemon = True
    self.p.start()


  def display_init(self):
    pangolin.CreateWindowAndBind('3D DISPLAY', self.W, self.H)
    gl.glEnable(gl.GL_DEPTH_TEST)

    self.scam = pangolin.OpenGlRenderState(pangolin.ProjectionMatrix(self.W, self.H,
                420, 420, self.W//2, self.H//2, 0.2, 100),
                pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))

    handler = pangolin.Handler3D(self.scam)
    self.dcam = pangolin.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    self.dcam.SetHandler(handler)

  # TODO: follow the car from a top-down view
  def renderer_main(self, poses_q, path_q):
    print("[+] Initializing 3D Display ...")
    self.display_init()

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    self.dcam.Activate(self.scam)

    # Draw camera
    while not pangolin.ShouldQuit():
      while not path_q.empty():
        if not poses_q.empty():
          self.poses = poses_q.get()
        self.path = path_q.get()

      gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
      self.dcam.Activate(self.scam)

      if  self.path is not None:
        # draw car current pose
        if self.poses is not None:
          gl.glLineWidth(1)
          gl.glColor3f(0.0, 1.0, 0.0)
          #pangolin.DrawCamera(self.poses)
          pangolin.DrawBoxes(np.array([self.poses]), np.array([[1., 1., 1.]]))

        # draw path
        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(self.path)

      pangolin.FinishFrame()

  # TODO: draw current location as pose and the rest of the predicted path as red dots
  def draw(self, path, pose=None):
    if self.poses_q is None or self.path_q is None:
      return

    #poses = [pose]
    #for pose in path:
    #  poses.append(pose)
    #poses = np.stack(poses)
    #print(poses.shape)
    if pose is not None:
      self.poses_q.put(pose)
    self.path_q.put(path)

# TODO: render 2D frame alongside it's path
class Renderer2D:
  def __init__(self, w, h):
    self.W = w
    self.H = h
    self.poses = None
    self.poses_q = Queue()
    self.path_q = Queue()
    self.path = None
    self.poses = None
    self.p = Process(target=self.renderer_main, args=(self.poses_q, self.path_q,))
    #self.p.daemon = True
    #self.p.start()
