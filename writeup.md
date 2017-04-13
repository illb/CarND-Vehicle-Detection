##Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

###Source Files
* debug.py : sandbox for debugging
* debug_interative_window_search.ipynb : sandbox for window search debugging
* train.py : train data
* process.py : detection pipeline process
* movie.py : make the output video
* modules
  * data.py : data list functions
  * camera_calibration.py : camera calibration functions
  * feature.py : feature extraction functions

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

* The code for feature extraction functions are in the file called `feature.py`.  

* I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

* sample image
  * vehicle / non vehicle

![vehicle_1](./output_images/train_vehicle_001.png =64x)
![non_vehicle_1](./output_images/train_non_vehicle_001.png =64x)

* In the YCrCb color space image, extract hog feature by channel.
  * `get_hog_features` function in `feature.py` can extract hog features

  * vehicle sample (Y,Cr,Cb) / vehicle hog (Y,Cr,Cb)

![vehicle_channel_Y_1](./output_images/feature_channel_Y_train_vehicle_001.png =64x)
![vehicle_channel_Cr_1](./output_images/feature_channel_Cr_train_vehicle_001.png =64x)
![vehicle_channel_Cb_1](./output_images/feature_channel_Cb_train_vehicle_001.png =64x)
![vehicle_hog_Y_1](./output_images/feature_hog_Y_train_vehicle_001.png =64x)
![vehicle_hog_Cr_1](./output_images/feature_hog_Cr_train_vehicle_001.png =64x)
![vehicle_hog_Cb_1](./output_images/feature_hog_Cb_train_vehicle_001.png =64x)

  * non vehicle sample (Y,Cr,Cb) / non vehicle hog (Y,Cr,Cb)

![non_vehicle_channel_Y_1](./output_images/feature_channel_Y_train_non_vehicle_001.png =64x)
![non_vehicle_channel_Cr_1](./output_images/feature_channel_Cr_train_non_vehicle_001.png =64x)
![non_vehicle_channel_Cb_1](./output_images/feature_channel_Cb_train_non_vehicle_001.png =64x)
![non_vehicle_hog_Y_1](./output_images/feature_hog_Y_train_non_vehicle_001.png =64x)
![non_vehicle_hog_Cr_1](./output_images/feature_hog_Cr_train_non_vehicle_001.png =64x)
![non_vehicle_hog_Cb_1](./output_images/feature_hog_Cb_train_non_vehicle_001.png =64x)

####2. Explain how you settled on your final choice of HOG parameters.

* I used a hog vector in all directions
* to make subsampling easier, I set the cell size to 3 powers of 2
* parameters
  * orient : 9
  * pix_per_cell : 8
  * cell_per_block : 2

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

* normalize fit a per-column scaler
  * line 23 in `train.py`

* split 80% of train set and 20% of test set

* trained a linear SVM 
  * lines 42 ~ 45 in `train.py`

* added color histogram features
  * vehicle color histogram

![vehicle_color_histogram_1](./output_images/feature_color_histogram_vehicle_001.png =800x)

  * non vehicle color histogram

![non_vehicle_color_histogram_1](./output_images/feature_color_histogram_non_vehicle_001.png =800x)

* added spatial binning of color
  * vehicle spatially binned
![spatially_binned_train_vehicle](./output_images/feature_spatially_binned_train_vehicle_001.png =800x)
  
  * non vehicle spatially binned
![spatially_binned_train_non_vehicle](./output_images/feature_spatially_binned_train_non_vehicle_001.png =800x)

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

* 

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![test1](./test_images/test1.jpg =400x)
![test1](./output_images/test1.jpg =400x)

![test2](./test_images/test2.jpg =400x)
![test2](./test_images/test2.jpg =400x)

![test3](./test_images/test3.jpg =400x)
![test3](./test_images/test3.jpg =400x)


![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

* using another architecture  : https://www.slideshare.net/xavigiro/ssd-single-shot-multibox-detector
  * Faster RNN : https://github.com/smallcorgi/Faster-RCNN_TF
  * YOLO : https://pjreddie.com/darknet/yolo/
  * SSD : https://github.com/rykov8/ssd_keras
