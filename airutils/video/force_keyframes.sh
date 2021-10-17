#!/bin/bash
ffmpeg -i $1 -vcodec libx264 -x264-params keyint=$2:scenecut=0 -acodec copy "${1%.*}_keyframes_at_$2.mp4"