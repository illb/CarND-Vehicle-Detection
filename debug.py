import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#######################################
import data
import feature
import window_search as ws
import util
from util import rgb

######################################

def show_sample_car_data(car_data, not_car_data):
    print('Your function returned a count of',
          car_data.len, ' cars and',
          not_car_data.len, ' non-cars')
    print('of size: ', car_data.image_shape, ' and data type:',
          car_data.image_data_type)

    # Read in car / not-car images
    car_image = mpimg.imread(car_data.sample_image())
    notcar_image = mpimg.imread(not_car_data.sample_image())

    # Plot the examples
    # fig = plt.figure()
    plt.subplot(121)
    plt.imshow(rgb(car_image))
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(rgb(notcar_image))
    plt.title('Example Not-car Image')
    plt.show()

car_data, not_car_data = data.get_car_data()
# show_sample_car_data(car_data, not_car_data)

#######################################

def show_hog_features(img):
    features, hog_image = feature.get_hog_features(img, visualise=True)

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()

img = car_data.sample_image()
# show_hog_features(img)

#######################################
def show_slide_windows(img):
    windows = ws.slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                              xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    window_img = util.draw_boxes(img, windows, color=(0, 0, 255), thick=6)

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(rgb(img))
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(rgb(window_img))
    plt.title('Slide Windows')
    plt.show()

show_slide_windows(img)
