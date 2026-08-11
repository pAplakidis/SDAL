#!/usr/bin/env python3
import os

data_dir = "../collected_data"

REQUIRED_FILES = [
  "desires.npy",
  "frame_paths.npy",
  "poses.npy",
  "video.mp4"
]


if __name__ == "__main__":
  todo_crossroads = []
  for folder in sorted(os.listdir(data_dir)):
    base = data_dir + "/" + folder
    print("[+] Checking: " + base)
    for f in REQUIRED_FILES:
      if f in os.listdir(base):
        print(f, "exists")
      else:
        print(f, "missing")
        print("Deleting folder", base)
        os.system("rm -rf " + base)
        break
    if "crossroads.npy" not in os.listdir(base):
      todo_crossroads.append(folder)
    print()

  print("TODO => label crossroads for folders:")
  for f in todo_crossroads:
    print(f)
