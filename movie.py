import cv2

import data
import camera_calibration as cc

img_size = (1280, 720)
from moviepy.editor import VideoFileClip

######################################
# Distrotion Correction
objpoints, imgpoints = cc.load_corners()
mtx, dist = cc.calibrate_camera(objpoints, imgpoints, img_size)

def pipeline(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    undist = cc.undistort(img, mtx, dist)

    result = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)

    return result


def save_movie():
    for video_path in data.get_video_paths():
        video_clip = VideoFileClip(video_path)
        white_clip = video_clip.fl_image(pipeline)
        white_clip.write_videofile("output_" + video_path, audio=False)

save_movie()