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

bboxes = ws.find_cars(image, svc, X_scaler, params)

def show_find_cars():
    draw_img = np.copy(image)
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 0, 255), 6)

    plt.imshow(util.rgb(draw_img))
    plt.show()

# show_find_cars()

################################

def show_heat():
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = ws.add_heat(heat, bboxes)

    # Apply threshold to help remove false positives
    heat = ws.apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

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

show_heat()