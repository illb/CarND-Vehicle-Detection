import numpy as np
import cv2
import feature
import util
import data

ystart = 400
ystop = 656
scale = 1.0

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_car_bboxes(img, svc, X_scaler, params: data.ModelParams, debug=False, ystart=ystart, ystop=ystop, scale=scale):
    bboxes = []
    target_bboxes = []
    tosearch = util.to_color_space(img[ystart:ystop, :, :], params.color_space)
    tosearch = feature.normalize(tosearch)

    if scale != 1:
        imshape = tosearch.shape
        tosearch = cv2.resize(tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # Define blocks and steps as above
    nxblocks = (tosearch.shape[1] // params.hog_pix_per_cell) + 1
    nyblocks = (tosearch.shape[0] // params.hog_pix_per_cell) + 1
    # nfeat_per_block = params.hog_orient * params.hog_cell_per_block ** 2
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
            y_cell = yb * cells_per_step
            x_cell = xb * cells_per_step

            xleft = x_cell * params.hog_pix_per_cell
            ytop = y_cell * params.hog_pix_per_cell

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
                hog_feat1 = hog1[y_cell:y_cell + nblocks_per_window, x_cell:x_cell + nblocks_per_window].ravel()
                hog_feat2 = hog2[y_cell:y_cell + nblocks_per_window, x_cell:x_cell + nblocks_per_window].ravel()
                hog_feat3 = hog3[y_cell:y_cell + nblocks_per_window, x_cell:x_cell + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                img_features.append(hog_features)

            features = np.hstack(img_features).reshape(1, -1)

            # Scale features and make a prediction
            test_features = X_scaler.transform(features)
            test_prediction = svc.predict(test_features)

            box_x = np.int(xleft * scale)
            box_y = np.int(ytop * scale)
            box_window = np.int(window * scale)
            pt1 = (box_x, box_y + ystart)
            pt2 = (box_x + box_window, box_y + box_window + ystart)
            if debug:
                target_bboxes.append((pt1, pt2))
            if test_prediction == 1:
                bboxes.append((pt1, pt2))
    if debug:
        return bboxes, target_bboxes
    else:
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


scales = [1.25, 1.5, 1.75, 2.0, 3.0, 4.0]
def find_cars(undist, svc, X_scaler, params):
    bboxes = []
    for scale in scales:
        bboxes.extend(find_car_bboxes(undist, svc, X_scaler, params, scale=scale))

    heat = np.zeros_like(undist[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, bboxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 10)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    return heatmap, bboxes
