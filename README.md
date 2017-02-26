**Vehicle Detection Project**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

###Summary###

In this project we are attempting to detect other vehicles within a driving video stream. The vehicles are Identified through a combination of feature extraction and linear classifiers. It is designed to cover the basics of those algorithms so it's performance is not optimized at this time.

####Video Results####

[![Video of Performance](http://img.youtube.com/vi/fjs2ltGHCPw/0.jpg)](http://www.youtube.com/watch?v=fjs2ltGHCPw)

The above video shows both the lane detection algorithm from [Advanced Lane Detection](https://github.com/johncarpenter/CarND-Vehicle-Detection) project and the VehicleDetection program.


###Analysis and Discovery###

The goals / steps of this project are the following:

1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
2. Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
3. Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
4. Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
5. Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
6. Estimate a bounding box for vehicles detected.

###Histogram of Oriented Gradients (HOG) Feature Extraction and SVM Classifier###

The first step in the project is to create a classifier model to be able to identify vehicles within an image. To do this we convert training images into a list of features. We then created a classifier to identify whether an object is a "car" or "not a car" using the feature sets. The images provided ( [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) ) were provided as part of the course materials

####Training images####
Provided are two sample images from the training set
[Car training image](output_images/training_car_sample.jpg)
[Not Car training image](output_images/training_non_car_sample.jpg)

After a number of iterations we arrived at a feature set that was made up of *HOG features + Spatial Binning + Color Histogram* Below is a section from the ```training.log``` file that was used in the model generation
```
Color Space HLS
Spatial Bin Size (16, 16)
Histogram Bins 16
Orientations 9
Px per Cell 16
Cell per Block 2
HOG Channel ALL
Use Spatial Features True
Use Histogram Features True
Use HOG Features True
Feature vector length: 1788
```

In order to choose the parameters the ```train.py``` program was run a number of times. This program trained the model and tested it on a series of 6 images for testing. The output of the program gave both the training parameters in the ```training.log``` file and a visual representation of the vehicle identification. This gave two measures to verify the classifier would be suitable for our video

```
usage: train.py [-h] cars notcars

Vehicle Detection Training Program

positional arguments:
  cars        Cars image directory
  notcars     Not cars image directory

optional arguments:
  -h, --help  show this help message and exit
  ```
In training the classifier, we separated 20% of the data for testing purposes. This gave a small measure for the
accuracy for the ```LinearSVC``` classifier. Below is the output from the training using the configuration parameters from above;

```
training.log

Training...
Color Space HLS
Spatial Bin Size (16, 16)
Histogram Bins 16
Orientations 9
Px per Cell 16
Cell per Block 2
HOG Channel ALL
Use Spatial Features True
Use Histogram Features True
Use HOG Features True
Training with 8792 Car features
Training with 8968 Non-Car features
Using: 9 orientations 16 pixels per cell and 2 cells per block
Feature vector length: 1788
3.85 Seconds to train SVC...
Test Accuracy of SVC =  0.9907
Training finished
```
Once the model was validated it was saved to ```vehicle_detection.p``` pickle file along with the configuration.

###Searching the Image###

Once the classifier was completed, the next step was to search the image for potential matches from the classifier. This was accomplished by creating sliding a test window across the image. Different window scales were required to handle different sized vehicles in the image. And so we repeated the search with a variety of window sizes.

![Sliding Windows](output_images/training_car_sample.jpg)
Shows the windows that were used within the image

The sliding windows proved to be the largest performance drain on the application so the processes to choosing the number of sliding windows was based on trying to minimize the search to as few windows as possible. The following steps were taken;
1. Minimize the search to just the area on the road. As seen in the image above
2. Maintain a historical record of the previous N iterations and use that to supplement the search in the current windows
3. Keep the number of different sizes to a minimum (2)






Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :).

You can submit your writeup in markdown or use another method and submit a pdf instead.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!
