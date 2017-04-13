import cv2
import numpy as np
import window_search as ws

import data
import camera_calibration as cc

img_size = (1280, 720)
from moviepy.editor import VideoFileClip

######################################
# Distrotion Correction
objpoints, imgpoints = cc.load_corners()
mtx, dist = cc.calibrate_camera(objpoints, imgpoints, img_size)

svc, X_scaler, params = data.load_model()

def pipeline(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    undist = cc.undistort(img, mtx, dist)

    heatmap = ws.find_cars(undist, svc, X_scaler, params)

    from scipy.ndimage.measurements import label

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    draw_img = ws.draw_labeled_bboxes(np.copy(undist), labels)

    result = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

    return result


def save_movie():
    for video_path in data.get_video_paths():
        video_clip = VideoFileClip(video_path)
        white_clip = video_clip.fl_image(pipeline)
        white_clip.write_videofile("output_" + video_path, audio=False)

save_movie()