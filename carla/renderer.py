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
    self.q = Queue()
    self.p = Process(target=self.renderer_main, args=(self.q,))
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

  def renderer_main(self, q):
    print("[+] Initializing 3D Display ...")
    self.display_init()

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    self.dcam.Activate(self.scam)

    # TODO: render poses instead of points
    # Draw camera
    while not pangolin.ShouldQuit():
      while not q.empty():
        self.poses = q.get()

      gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
      self.dcam.Activate(self.scam)

      if self.poses is not None:
        #gl.glLineWidth(1)
        #gl.glColor3f(0.0, 1.0, 0.0)
        #pangolin.DrawCameras(self.poses, 0.5, 0.75, 0.8)
        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(self.poses)

      pangolin.FinishFrame()

  def draw(self, path):
    if self.q is None:
      return

    poses = []
    for pose in path:
      poses.append(pose)
    self.q.put(np.array(poses))
