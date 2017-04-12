import cv2
import numpy as np
import matplotlib.pyplot as plt

import data, util
import camera_calibration as cc
import window_search as ws

img_size = (1280, 720)
output_dir = "./output_images"

######################################
print("========= 01) load model")

svc, X_scaler, params = data.load_model()

######################################
print("========= 02) undist images")

objpoints, imgpoints = cc.load_corners()
mtx, dist = cc.calibrate_camera(objpoints, imgpoints, img_size)
image = util.imread('test_images/test3.jpg')
image = cc.undistort(image, mtx, dist)

######################################

windows = ws.slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                          xy_window=(96, 96), xy_overlap=(0.5, 0.5))

hot_windows = ws.search_windows(image, windows, svc, X_scaler, params)

def show_serach_windows():
    draw_image = np.copy(image)
    window_img = util.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    plt.imshow(util.rgb(window_img))
    plt.title('search_windows')
    plt.show()

# show_serach_windows()

################################


import random
def _rand256():
    return random.randrange(0, 256)

ystart = 400
ystop = 656
scale = 1.5

bboxes, target_bboxes = ws.find_car_bboxes(image, svc, X_scaler, params, debug=True,
                                           ystart=ystart, ystop=ystop, scale=scale)

def show_find_cars():
    draw_img = np.copy(image)
    for bbox in target_bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], (_rand256(), _rand256(), _rand256()), 2)
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 0, 255), 6)
    cv2.rectangle(draw_img, (0, ystart), (img_size[0], ystop), (255, 0, 0), 6)

    plt.imshow(util.rgb(draw_img))
    plt.show()

show_find_cars()

################################

def show_heat():
    heatmap = ws.find_car_map(image, svc, X_scaler, params)

    from scipy.ndimage.measurements import label

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    draw_img = ws.draw_labeled_bboxes(np.copy(image), labels)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(util.rgb(draw_img))
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    plt.show()

#show_heat()