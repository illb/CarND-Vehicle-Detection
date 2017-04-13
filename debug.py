import matplotlib.pyplot as plt
import data
import numpy as np
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
    car_image = car_data.sample_image()
    notcar_image = not_car_data.sample_image()

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

# Define a function that takes an image,
# start and stop positions in both x and y, window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    x_span = x_start_stop[1] - x_start_stop[0]
    y_span = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    window_w = xy_window[0]
    window_h = xy_window[1]
    x_pixels_per_step = np.int(window_w * (1.0 - xy_overlap[0]))
    y_pixels_per_step = np.int(window_h * (1.0 - xy_overlap[1]))

    # Compute the number of windows in x/y
    x_overlap_buffer = np.int(window_w * (xy_overlap[0]))
    y_overlap_buffer = np.int(window_h * (xy_overlap[1]))
    x_window_count = np.int((x_span - x_overlap_buffer) / x_pixels_per_step)
    y_window_count = np.int((y_span - y_overlap_buffer) / y_pixels_per_step)

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    for y_window_index in range(y_window_count):
        for x_window_index in range(x_window_count):
            # Calculate window position
            start_x = x_window_index * x_pixels_per_step + x_start_stop[0]
            end_x = start_x + window_w
            start_y = y_window_index * y_pixels_per_step + y_start_stop[0]
            end_y = start_y + window_h
            # Append window position to list
            window_list.append(((start_x, start_y), (end_x, end_y)))
    # Return the list of windows
    return window_list


def show_slide_windows(img):
    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
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

# img = util.imread(data.get_test_paths()[0])
# show_slide_windows(img)
