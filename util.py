import numpy as np
import cv2

def imread(path):
    return cv2.imread(path)

def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def to_color_space(image, color_space = 'RGB'):
    result = None
    if color_space != 'BGR':
        if color_space == 'RGB':
            result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space == 'HSV':
            result = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            result = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            result = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            result = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            result = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        result = np.copy(image)
    return result

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy