#!/usr/bin/env bash

bash reinstall.sh
# fix cv2 import
apt-get update && apt-get install -y libsndfile1 ffmpeg
# install LDM-related dependencies
pip install kornia==0.6
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
