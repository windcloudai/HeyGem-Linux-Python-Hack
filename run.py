import argparse
import gc
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import uuid
from enum import Enum

import cv2
from flask import Flask, request

if sys.version_info.major != 3 or sys.version_info.minor != 8:
    print("请使用 Python 3.8 版本运行此脚本")
    sys.exit(1)

import service.trans_dh_service
from h_utils.custom import CustomError
from y_utils.config import GlobalConfig
from y_utils.logger import logger


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=(argparse.ArgumentDefaultsHelpFormatter)
    )

    parser.add_argument(
        "--audio_path",
        type=str,
        default="example/audio.wav",
        help="path to local audio file",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="example/video.mp4",
        help="path to local video file",
    )
    opt = parser.parse_args()
    return opt





def main():
    opt = get_args()
    if not os.path.exists(opt.audio_path):
        audio_url = "example/audio.wav"
    else:
        audio_url = opt.audio_path

    if not os.path.exists(opt.video_path):
        video_url = "example/video.mp4"
    else:
        video_url = opt.video_path
    sys.argv = [sys.argv[0]]
    task = service.trans_dh_service.TransDhTask()

    code = "1004"
    task.work(audio_url, video_url, code, 0, 0, 0, 0)


if __name__ == "__main__":
    main()

# python run.py
# python run.py --audio_path example/audio.wav --video_path example/video.mp4
