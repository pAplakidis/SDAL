#!/bin/bash

read -p "Enter starting dataset index: " start
read -p "Enter the ending dataset index: " end

for((i = start;i <= end; i++)); do
  export DATA_PATH="../collected_data/$i/"
  export RENDER="False"
  echo ""
  echo "[+++] Running replay for dataset $i ..."
  echo ""
  ./replay.py
done


