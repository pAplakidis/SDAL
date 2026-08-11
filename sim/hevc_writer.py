import shutil
import subprocess

class HEVCWriter:
  def __init__(self, path, width, height, fps, encoder="libx265", crf=23, preset="medium"):
    if shutil.which("ffmpeg") is None:
      raise RuntimeError("ffmpeg not found. Install it with: sudo apt install ffmpeg")

    command = [
      "ffmpeg",
      "-y",

      # Input: raw OpenCV-style BGR frames
      "-f", "rawvideo",
      "-pix_fmt", "bgr24",
      "-s:v", f"{width}x{height}",
      "-r", str(fps),
      "-i", "-",

      "-an",

      # H.265 / HEVC
      "-c:v", encoder,
    ]

    if encoder == "libx265":
      command += [
        "-preset", preset,
        "-crf", str(crf),
      ]
    elif encoder == "hevc_nvenc":
      command += [
        "-preset", "p4",
        "-cq", str(crf),
        "-b:v", "0",
      ]

    command += [
      "-pix_fmt", "yuv420p",

      # Raw HEVC elementary stream, comma-style.
      "-f", "hevc",
      path,
    ]

    print("[*] Starting HEVC encoder:")
    print(" ".join(command))
    self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

  def write(self, frame):
    if self.process.poll() is not None:
      raise RuntimeError(f"FFmpeg exited unexpectedly with code {self.process.returncode}")

    try:
      self.process.stdin.write(frame.tobytes())
    except BrokenPipeError:
      raise RuntimeError("FFmpeg HEVC encoder closed its input pipe")

  def close(self):
    if self.process is None:
      return

    if self.process.stdin is not None:
      self.process.stdin.close()

    self.process.wait()

    if self.process.returncode != 0:
      raise RuntimeError(f"FFmpeg exited with code {self.process.returncode}")
