import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import data, util, feature
import camera_calibration as cc
import window_search as ws

img_size = (1280, 720)
output_dir = "./output_images"

vehicle = "./output_images/train_vehicle_001.png"
non_vehicle = "./output_images/train_non_vehicle_001.png"

def float_to_gray(img):
    return (img * 255).astype(np.int32)


#######################################

def save_features(path):
    img = util.imread(path)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    def save_channel_features(idx, channel_name):
        channel = ycrcb[:, :, idx]
        features, hog_image = feature.get_hog_features(channel, vis=True)
        cv2.imwrite(output_dir + "/feature_channel_{}_".format(channel_name) + os.path.basename(path), channel)
        cv2.imwrite(output_dir + "/feature_hog_{}_".format(channel_name) + os.path.basename(path), float_to_gray(hog_image))

    save_channel_features(0, "Y")
    save_channel_features(1, "Cr")
    save_channel_features(2, "Cb")

# save_features(vehicle)
# save_features(non_vehicle)

#######################################

def show_color_histogram(path):
    img = util.imread(path)
    nbins = 32
    bins_range = (0, 256)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    channel1_hist = np.histogram(ycrcb[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(ycrcb[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(ycrcb[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = channel1_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

    # Plot a figure with all three bar charts
    fig = plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.bar(bin_centers, channel1_hist[0])
    plt.xlim(0, 256)
    plt.title('Y Histogram')
    plt.subplot(132)
    plt.bar(bin_centers, channel2_hist[0])
    plt.xlim(0, 256)
    plt.title('Cr Histogram')
    plt.subplot(133)
    plt.bar(bin_centers, channel3_hist[0])
    plt.xlim(0, 256)
    plt.title('Cb Histogram')
    plt.show()

#show_color_histogram(vehicle)
#show_color_histogram(non_vehicle)

#######################################

def show_spatial_binning(path):
    img = util.imread(path)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    feature_vec = feature.bin_spatial(ycrcb, size=(32, 32))

    # Plot features
    plt.plot(feature_vec)
    plt.title('Spatially Binned Features')
    plt.show()

# show_spatial_binning(vehicle)
#show_spatial_binning(non_vehicle)


######################################
svc, X_scaler, params = data.load_model()
objpoints, imgpoints = cc.load_corners()
mtx, dist = cc.calibrate_camera(objpoints, imgpoints, img_size)

######################################

import random
def _rand256():
    return random.randrange(0, 256)

def save_windows():
    for i in range(6):
        path = "./test_images/test{}.jpg".format(i+1)
        img = util.imread(path)
        undist = cc.undistort(img, mtx, dist)

        scale = 1.0
        bboxes, target_bboxes = ws.find_car_bboxes(undist, svc, X_scaler, params, debug=True,
                                                   scale=1.0)
        draw_img = np.copy(undist)
        for bbox in target_bboxes:
            cv2.rectangle(draw_img, bbox[0], bbox[1], (_rand256(), _rand256(), _rand256()), 2)
        for bbox in bboxes:
            cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 0, 255), 6)

        cv2.imwrite(output_dir + "/window_scale_{}_".format(scale) + os.path.basename(path), draw_img)


def save_window_searches():
    for i in range(6):
        path = "./test_images/test{}.jpg".format(i+1)
        img = util.imread(path)
        undist = cc.undistort(img, mtx, dist)

        heatmap, bboxes = ws.find_cars(undist, svc, X_scaler, params)

        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        plt.savefig(output_dir + "/window_heatmap_test{}.png".format(i+1))

        from scipy.ndimage.measurements import label

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        draw_img = ws.draw_labeled_bboxes(np.copy(undist), labels)
        cv2.imwrite(output_dir + "/window_result_" + os.path.basename(path), draw_img)

        draw_img = np.copy(undist)
        for bbox in bboxes:
            cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 0, 255), 4)

        cv2.imwrite(output_dir + "/window_bboxes_" + os.path.basename(path), draw_img)

save_window_searches()
