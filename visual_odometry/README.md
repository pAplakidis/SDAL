# Visual Odometry Report

Canonical VO report. Workflow/setup now live in `sim/README.md`.

## Goal

`sim/post_process_clip.py` now adds visual odometry as another pose source for comparison with ground truth, wheel odometry, IMU, and GNSS. Visual odometry runs realtime, frame by frame, from RGB frames and outputs only poses; it does not build a point map, run bundle adjustment, or perform loop closure.

## Implementation

The implementation lives in `visual_odometry/monocular.py` and is wired into `sim/post_process_clip.py`.

Pipeline:

1. Read frames from `video.hevc`.
2. Optionally downscale frames with `VO_IMAGE_SCALE` for speed.
3. Detect corners with `cv2.goodFeaturesToTrack`.
4. Compute ORB descriptors at those points.
5. Match consecutive frames with Hamming BF matching and Lowe ratio filtering.
6. Estimate essential matrix with `cv2.findEssentialMat` and RANSAC.
7. Recover relative rotation and translation direction with `cv2.recoverPose`.
8. Scale translation by `speed_mps * ODOMETRY_DT`.
9. Accumulate relative transforms into pose stream.
10. Convert OpenCV camera axes to local vehicle-like axes.
11. Draw detected keypoints on 2D frame during replay.

Output pose schema matches other sources:

```text
[x y z roll pitch yaw]
```

## Coordinate Frames

OpenCV camera coordinates are converted to local pose coordinates before display:

```text
OpenCV x right  -> local y right
OpenCV y down   -> local z up, negated
OpenCV z forward -> local x forward
```

The resulting poses are zero-started and normalized by existing `post_process_clip.py` helpers before Open3D display.

## Scale

This is monocular visual odometry, so translation from `recoverPose` has arbitrary scale. Current implementation uses CARLA speed to get useful metric motion:

```text
translation_scale = speeds[frame_idx - 1] * ODOMETRY_DT
```

This makes visual output comparable with existing CARLA pose sources. For real-world data, VO can still run without speed, but translation scale becomes unitless unless another source supplies scale later.

## Display

Visual odometry is drawn as cyan in 3D view and keypoints are drawn on 2D frame:

```python
color=(0.0, 1.0, 1.0)
stream_id="visual_odometry"
```

Existing colors:

```text
ground truth: green
odometry: blue
IMU: magenta
GNSS: yellow
visual odometry: cyan
```

## Tuning

Visual odometry tuning values are script constants in `sim/post_process_clip.py`, not environment variables:

```python
VO_ENABLED = True
VO_IMAGE_SCALE = 0.5
VO_MAX_FEATURES = 2000
VO_MATCH_RATIO = 0.75
VO_MAX_HAMMING_DISTANCE = 32
VO_MIN_INLIERS = 20
VO_RANSAC_THRESHOLD = 1.0
```

Edit the script directly when tuning these values.

VO is computed realtime inside replay loop, so tuning changes apply on next run.

## Limitations

This is pose-only frame-to-frame VO, not full SLAM.

Known limitations:

1. No loop closure.
2. No bundle adjustment.
3. No keyframe management.
4. No point map persistence.
5. No robust scale recovery without speed, stereo, depth, GNSS, or IMU fusion.
6. Dynamic traffic and low-texture road scenes can reduce match quality.

## Next Steps

Planned sensor fusion can use this visual pose stream with IMU, GNSS, odometry, and ground truth for Kalman filtering and error comparison.
