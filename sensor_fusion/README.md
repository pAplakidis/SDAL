# Sensor Fusion Report

`sensor_fusion/ekf.py` fuses sensor pose estimates into `predicted_poses.npy` using an extended Kalman filter.

## State

EKF state:

```text
[x, y, z, roll, pitch, yaw, vx, vy, vz]
```

Angles are radians inside filter and degrees in saved pose output.

## Prediction

Prediction uses:

- wheel odometry speed
- steering angle bicycle model
- IMU gyroscope
- IMU accelerometer with gravity compensation

The yaw-rate prediction blends steering-derived yaw rate and IMU gyro yaw rate.

## Corrections

Measurements used for correction:

- GNSS translation: `[x, y, z]`
- visual odometry pose: `[x, y, z, roll, pitch, yaw]`

Ground truth is never used by the EKF. It is only used for metrics.

## Outputs

`sim/post_process_clip.py` writes:

```text
predicted_poses.npy
pose_metrics.json
```

Predicted poses are rendered in red in the 3D display.

## Metrics

Metrics compare `predicted_poses` against zero-started ground truth:

- translation RMSE
- translation MAE
- max translation error
- final translation error
- per-axis RMSE
- roll/pitch/yaw RMSE

## Current Limits

Noise values are fixed constants in `PoseEKF`. They should be tuned per simulator or real-world dataset. Visual odometry remains monocular, so scale still depends on speed input unless another sensor provides scale.
