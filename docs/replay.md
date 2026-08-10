# CARLA Replay

`carla/replay.py` replays one collected CARLA dataset. It loads raw arrays, decodes `video.hevc`, colorizes `segmentation.npy`, renders normalized poses with Open3D, and overlays future path points on the RGB video.

## Inputs

Dataset directory defaults to `data/1` and can be overridden with `DATA_PATH`.

Required files:

- `video.hevc`: RGB dashcam recording.
- `segmentation.npy`: raw CARLA semantic label IDs with shape `(frames, height, width)`.
- `poses.npy`: raw vehicle poses with shape `(frames, 6)`, ordered `[x, y, z, roll, pitch, yaw]`.
- `desires.npy`: per-frame desire IDs.
- `steering_angles.npy`: per-frame steering values.
- `speeds.npy`: per-frame speed values.
- `imu.npy`: per-frame IMU values.
- `gnss.npy`: per-frame GNSS values.

## Displays

- `CARLA HEVC`: RGB video with frame metadata and projected future path.
- `CARLA Segmentation`: `segmentation.npy` colorized with CARLA CityScapes colors.
- `Display 3D`: Open3D trajectory and camera frustums.

## Path Projection

Future `poses.npy` positions are projected onto the RGB image only. The projection uses the current vehicle pose as camera reference, the collector camera offset `[0.8, 0.0, 1.13]`, and the CARLA RGB camera FOV of `70` degrees.

Projection flow:

1. Select future path points from `frame_idx` to `frame_idx + LOOKAHEAD`.
2. Transform world points into the current camera frame.
3. Convert CARLA camera coordinates to OpenCV image coordinates.
4. Drop points behind the camera or outside the frame.
5. Draw a magenta polyline and yellow/orange points on the RGB frame.

## Controls

Run with:

```bash
DATA_PATH=/home/pavlos/Dev/SDAL/data/1 ~/venv/bin/python carla/replay.py
```

Environment switches:

- `RENDER_3D=0`: disable Open3D renderer.
- `RENDER_2D=0`: disable OpenCV windows.
- `MAX_FRAMES=10`: limit replay length.
- `LOOKAHEAD=200`: number of future frames projected onto RGB image.

Press `q` in an OpenCV window to stop replay.

## Notes

- 3D pose display is normalized separately from image projection. Normalized poses subtract the first pose and auto-fit the path to Open3D scale.
- Image projection uses raw CARLA coordinates, not normalized Open3D coordinates.
- If no path appears, future points may be behind the current camera, outside the camera FOV, or hidden by inaccurate pose/camera calibration.
