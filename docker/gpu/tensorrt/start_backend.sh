#!/bin/bash
set -e

# RUN tensorrt backend
source "modularmot_venv/bin/activate" && \
rosrun rasberry_perception detection_server.py --backend tensorrt --config_path config.json --service_name $SERVICE_NAME
