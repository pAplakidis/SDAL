#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${SCRIPT_DIR}/../data"
STARTING_MAP=0
LAST_MAP=6
RUN_POST_PROCESS=1
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

mkdir -p "$OUT_ROOT"

# make dataset
# 1 min of each MAP for every WEATHER and TIME-OF-DAY scenarios
for ((map=STARTING_MAP; map <= LAST_MAP; map++)); do
  for weather in "${weathers[@]}"; do
    OUT_PATH="$OUT_ROOT/$dataset_idx/"
    mkdir -p "$OUT_PATH"

    echo ""
    echo "[+++] Collecting: MAP=$map WEATHER=$weather dataset_idx=$dataset_idx OUT_PATH=$OUT_PATH"
    echo ""

    OUT_PATH="$OUT_PATH" MAP="$map" WEATHER="$weather" RENDER=0 python "$SCRIPT_DIR/carla_collector.py"

    if [[ "$RUN_POST_PROCESS" == "1" ]]; then
      echo "[+++] Post-process: dataset_idx=$dataset_idx"
      RENDER=0 DATA_PATH="$OUT_PATH" python "$SCRIPT_DIR/post_process_clip.py"
    fi

    ((dataset_idx++))
  done
done
