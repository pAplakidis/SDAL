# SDAL Workflow

## Setup

1. Create/activate venv.
2. Install base deps:

```bash
python -m pip install -r requirements.txt
```

3. Install extra runtime deps used by replay/post-process as needed:

```bash
python -m pip install open3d scipy scikit-image
```

4. Install CARLA 0.9.16 Python API/egg and start CARLA server.
5. Run commands from `sim/` unless noted otherwise.

## Data Collection

Start CARLA server:

```bash
./start_carla.sh
```

Collect dataset:

```bash
OUT_PATH="../collected_data/1/" MAP=0 WEATHER=CloudyNoon RENDER=0 ./carla_collector.py
```

Collector writes:

- `video.hevc`
- `poses.npy`
- `desires.npy`
- `steering_angles.npy`
- `throttles.npy`
- `speeds.npy`
- `imu.npy`
- `gnss.npy`
- `segmentation.npy`

## Replay

Replay dataset:

```bash
DATA_PATH="../collected_data/1/" RENDER=False ./replay.py
```

Replay shows:

- RGB video
- semantic segmentation
- 3D normalized ground-truth path
- projected future path on RGB frame

## Post Process

Post-process clip with wheel odometry, IMU, GNSS, and visual odometry:

```bash
DATA_PATH="../collected_data/1/" python post_process_clip.py
```

This script:

- replays RGB video
- draws GT, odometry, IMU, GNSS, and visual odometry in 3D
- overlays visual odometry keypoints on 2D frame

## Workflow

1. Start CARLA server.
2. Collect dataset with `carla_collector.py`.
3. Replay raw data with `replay.py`.
4. Run `post_process_clip.py` for pose comparisons and visual odometry.

## Notes

- `carla_collector.py` expects `RENDER=0` or `RENDER=1`.
- `replay.py` accepts string boolean `RENDER=False`/unset.
- `post_process_clip.py` currently accepts standard string boolean values for `RENDER`.
- Visual odometry lives in `../visual_odometry/monocular.py` and runs frame-by-frame.
