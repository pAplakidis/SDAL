# Self Driving Auto-Labeler

This is a custom auto-ground-truthing stack utilizing deep learning (segmentation), visual odometry, sensor data and Kalman filter to generate a path and road edges
It uses data from Carla autonomous driving simulator

## Sensors to extract data from:
- Gyroscope/IMU
- GPS/GNSS
(the 2 preceding sensors are not needed for simulation data, but for real world)
- camera => visual odometry
- steering wheel angle sensor

## Libraries used (or to be used in the future):
- Laika for GNSS
- Rednose for Kalman
- SegNet for segmentation


## TODO:
* optimize semantic segmentation
* extract labels from segmented images using digital image processing
* detect poses instead of points (Rt matrix)  (optional)
* implement visual SLAM
* implement sensor fusion (visual odometry, GNSS, IMU) using Kalman Filters
* implement a localizer

