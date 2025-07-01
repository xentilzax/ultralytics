#!/bin/bash
mkdir -p datasets
cp -r source_dataset/* datasets/

xhost +local:docker && \
docker run \
  -e DISPLAY=$DISPLAY \
  -it \
  --ipc=host \
  --gpus all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/.Xauthority:/root/.Xauthority \
  -v $PWD:/workspace \
  -v $PWD/result:/ultralytics/runs/detect/ \
  -it --ipc=host ultralytics/ultralytics:latest \
  yolo detect train \
  data=/workspace/datasets/data.yaml \
  model=/workspace/datasets/yolo11.yaml \
  pretrained=/ultralytics/yolo11n.pt \
  batch=32 \
  epochs=100 \
  imgsz=360,640 \
  augment=False
