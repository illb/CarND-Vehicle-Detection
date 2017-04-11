import numpy as np
import cv2
import feature
import util
import data

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


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, params: data.ModelParams):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = feature.single_img_features(test_img, params)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows

ystart = 400
ystop = 656
scale = 1.5


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, svc, X_scaler, params: data.ModelParams, ystart=ystart, ystop=ystop, scale=scale):
    bboxes = []
    tosearch = util.to_color_space(img[ystart:ystop, :, :], params.color_space)
    tosearch = feature.normalize(tosearch)

    if scale != 1:
        imshape = tosearch.shape
        tosearch = cv2.resize(tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # Define blocks and steps as above
    nxblocks = (tosearch.shape[1] // params.hog_pix_per_cell) - 1
    nyblocks = (tosearch.shape[0] // params.hog_pix_per_cell) - 1
    nfeat_per_block = params.hog_orient * params.hog_cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // params.hog_pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    ch1 = tosearch[:, :, 0]
    ch2 = tosearch[:, :, 1]
    ch3 = tosearch[:, :, 2]
    hog1 = feature.get_hog_features(ch1, params.hog_orient, params.hog_pix_per_cell, params.hog_cell_per_block, feature_vec=False)
    hog2 = feature.get_hog_features(ch2, params.hog_orient, params.hog_pix_per_cell, params.hog_cell_per_block, feature_vec=False)
    hog3 = feature.get_hog_features(ch3, params.hog_orient, params.hog_pix_per_cell, params.hog_cell_per_block, feature_vec=False)
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            xleft = xpos * params.hog_pix_per_cell
            ytop = ypos * params.hog_pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            img_features = []
            if params.spatial_feat:
                spatial_features = feature.bin_spatial(subimg, size=params.spatial_size)
                img_features.append(spatial_features)

            if params.hist_feat:
                hist_features = feature.color_hist(subimg, nbins=params.hist_bins)
                img_features.append(hist_features)

            if params.hog_feat:
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                img_features.append(hog_features)

            features = np.hstack(img_features).reshape(1, -1)

            # Scale features and make a prediction
            test_features = X_scaler.transform(features)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                pt1 = (xbox_left, ytop_draw + ystart)
                pt2 = (xbox_left + win_draw, ytop_draw + win_draw + ystart)
                bboxes.append((pt1, pt2))

    return bboxes

##################################


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img