#!/bin/bash
starting_map=3  # CHANGE THIS
n_maps=7
weathers=(
  "ClearNoon"
  "CloudyNoon"
  "WetNoon"
  "WetCloudyNoon"
  "MidRainyNoon"
  "HardRainNoon"
  "SoftRainNoon"
  "ClearSunset"
  "CloudySunset"
  "WetSunset"
  "WetCloudySunset"
  "MidRainSunset"
  "HardRainSunset"
  "SoftRainSunset"
  "ClearNight"
  "CloudyNight"
  "HardRainNight"
  "SoftRainNight"
)

read -p "Enter starting dataset index: " dataset_idx

# make dataset
# 1 min of each MAP for every WEATHER and TIME-OF-DAY scenarios
for((map=starting_map; map <= $n_maps; map++)); do
  for weather in "${weathers[@]}"; do
    export OUT_PATH="../collected_data/$dataset_idx/"
    export MAP="$map"
    export WEATHER="$weather"
    export RENDER="False"

    echo ""
    echo "[+++] Running collector: MAP=$map WEATHER=$weather dataset_idx=$dataset_idx ..."
    echo ""
    ./carla_collector.py
    ((dataset_idx++))
  done
done
