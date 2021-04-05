## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png

[image8]: ./images/test1.jpg
[image9]: ./images/test2.jpg
[image10]: ./images/test3.jpg
[image11]: ./images/test4.jpg
[image12]: ./images/test5.jpg
[image13]: ./images/test6.jpg

[image14]: ./images/full_pipeline/test1.png
[image15]: ./images/full_pipeline/test2.png
[image16]: ./images/full_pipeline/test3.png
[image17]: ./images/full_pipeline/test4.png
[image18]: ./images/full_pipeline/test5.png
[image19]: ./images/full_pipeline/test6.png

[video1]: ./project_video_cars_1.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

All the rubrics points are covered in the writeup and in the code comments as well.

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!


I have implemented the vehicle detection code in the file "vehicle_detector.py".

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

This step is codded in between the lines 55 through 68 of the file called `vehicle_detector.py`).   

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and tried to judge the performance based on the human ability which is to differenciate vehicles from non-vehicles. This is done by simply looking at the HOG iamge representation. I have collected the well fitted parameters values at the initial stage and then moved to next step of training a classifire for classification of vehicles and non vehicles based on the extracted features. after training looking at the training accuracy, parameters are further tuned.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in the file `vehicle_detector.py` specifically in lines 87 through 94, with different parameter combinations. After monitoring the accuracy, the Parameters are tuned to the following values.

|Parameter |--------->| value|
|----------||--------|
|color space||'YCrCb'|
|HOG orientations||9|
|HOG pixels per cell||8|
|HOG cells per block||2|
|HOG channel||"ALL"|
|Spatial binning dimensions||(16, 16)|
|Number of histogram bins||16|
|Spatial features on or off||True|
|Histogram features on or off||True|
|HOG features on or off||True|



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have used three different window sizes for detecting a vehicle according to the distance of vehicle in the frame. Window size 128px is used to detect the new vehicles i=entering into the frame from the sides. The window size of 96px is used to detect vehicles in the range near to car and across the image as well. 64px window is used to detect far range of the vehicles.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_cars_1.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In the lines 181 through 203 in the file `search_classify.py`, the positions of positive detections in each frame of the video are reordered.  From the positive detections a heatmap is created and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

#### Challenges
* Finding the most efficient feature vector for training the classifier and also using it to predict the image windows is the hardest part. In addition to that finding the parameters for the HOG features which matches the best is also challenging, as it is difficult to find the balance between the number of features and prediction accuracy.
* False positives are handled by the heat map. But the false negatives are the big weakness for this system, especially when we consider the vehicles far away range.

#### Improvements 
* the improved tracking algorithm should be used to improve the detection quality. Kalman filter would be great approach. when we first predict a vehicle, we initialize the estimate. we predict the location and a dimensions for the created estimates in the next frame. Every observation is associated with one of the previous esimate. we use this previous knowledge for smoothing out the fluctuations in the dimension and the position.while detection.


  

