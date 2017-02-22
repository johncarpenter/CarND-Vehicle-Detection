import matplotlib.image as mpimg

import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import sys

from utils import *
from Config import Config
from SearchConfig import SearchConfig

class VehicleDetection:
    '''
    Parameters:
    config - Config.py object
    '''
    def __init__(self,config = None, from_pickle=None):
        self.config = config
        self.svc = None
        self.xscaler = None
        self.has_trained = False
        if(from_pickle):
            self.__load_from_pickle(filename=from_pickle)

    def __load_from_pickle(self,filename="vehicle_detection.p"):
        #TODO handle missing file or errors
        dist_pickle = pickle.load( open( filename, "rb" ) )
        self.svc = dist_pickle["svc"]
        self.xscaler = dist_pickle["xscaler"]
        self.config = dist_pickle["config"]
        self.has_trained = True


    def train(self, car_image_directory, not_car_image_directory):

        # Output the current config
        self.config.dump()

        cars = glob.glob(car_image_directory+'/**/*.png', recursive=True)
        print("Training with {} Car features".format(len(cars)))
        not_cars = glob.glob(not_car_image_directory+'/**/*.png', recursive=True)
        print("Training with {} Non-Car features".format(len(not_cars)))


        car_features = extract_features(cars, color_space=self.config.color_space,
                                spatial_size=self.config.spatial_size, hist_bins=self.config.hist_bins,
                                orient=self.config.orient, pix_per_cell=self.config.pix_per_cell,
                                cell_per_block=self.config.cell_per_block,
                                hog_channel=self.config.hog_channel, spatial_feat=self.config.spatial_feat,
                                hist_feat=self.config.hist_feat, hog_feat=self.config.hog_feat)
        notcar_features = extract_features(not_cars, color_space=self.config.color_space,
                                spatial_size=self.config.spatial_size, hist_bins=self.config.hist_bins,
                                orient=self.config.orient, pix_per_cell=self.config.pix_per_cell,
                                cell_per_block=self.config.cell_per_block,
                                hog_channel=self.config.hog_channel, spatial_feat=self.config.spatial_feat,
                                hist_feat=self.config.hist_feat, hog_feat=self.config.hog_feat)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        self.xscaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = self.xscaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:',self.config.orient,'orientations',self.config.pix_per_cell,
            'pixels per cell and', self.config.cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        # Save the Training File
        dist_pickle = {}
        dist_pickle["svc"] = self.svc
        dist_pickle["xscaler"] = self.xscaler
        dist_pickle["config"] = self.config
        pickle.dump( dist_pickle, open( "vehicle_detection.p", "wb" ) )
        self.has_trained = True

    def process_image(self,img):

        if(self.has_trained == False):
            sys.exit("Config or training file not provided. Exiting")

        # if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        if(img.max() > 1):
            img = img.astype(np.float32)/255


        image_features = self.__single_image_features(img)

        x_start_stop = [None,None]
        x_start_stop[0] = 400
        x_start_stop[1] = img.shape[1]

        y_start_stop = [None,None]
        y_start_stop[0] = int(img.shape[0]/2)
        y_start_stop[1] = img.shape[0]-100

        hot_windows = []

        hot_windows += self.__search(img,SearchConfig(y_start_stop=y_start_stop, xy_window=(80,80),xy_overlap=(0.75,0.75)))
        hot_windows += self.__search(img,SearchConfig(y_start_stop=y_start_stop, xy_window=(128,128),xy_overlap=(0.75,0.75)))
        #hot_windows += self.__search(img,SearchConfig(y_start_stop=y_start_stop, xy_window=(184,184),xy_overlap=(0.75,0.75)))
        hot_windows += self.__search(img,SearchConfig(y_start_stop=y_start_stop, xy_window=(220,220),xy_overlap=(0.75,0.75)))

        #window_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)
        heatmap = filter_by_heatmap(img, hot_windows,thresh=9)
        labels = label(heatmap)

        draw_img = draw_labeled_bboxes(img, labels)
        return draw_img


    def __search(self,img, search_config):

        windows = slide_window(img, y_start_stop=search_config.y_start_stop, x_start_stop=search_config.x_start_stop,
                            xy_window=search_config.xy_window,xy_overlap=search_config.xy_overlap)
        return self.__search_windows(img, windows)


    def __single_image_features(self,img):
        #1) Define an empty list to receive features
        img_features = []
        #2) Apply color conversion if other than 'RGB'
        feature_image = convert_image(img,color_space=self.config.color_space)
        #3) Compute spatial features if flag is set
        if self.config.spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=self.config.spatial_size)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if self.config.hist_feat == True:
            hist_features = color_hist(feature_image, nbins=self.config.hist_bins, bins_range=(0,1))
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if self.config.hog_feat == True:
            if self.config.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                        self.config.orient, self.config.pix_per_cell, self.config.cell_per_block,
                                        vis=False, feature_vec=True))
            else:
                hog_features = get_hog_features(feature_image[:,:,self.config.hog_channel], self.config.orient,
                            self.config.pix_per_cell, self.config.cell_per_block, vis=False, feature_vec=True)
            #8) Append features to list
            img_features.append(hog_features)

        #9) Return concatenated array of features
        return np.concatenate(img_features)


    def __search_windows(self,img, windows):

        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            #4) Extract features for that window using single_img_features()
            features = self.__single_image_features(test_img)
            #5) Scale extracted features to be fed to classifier
            test_features = self.xscaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = self.svc.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows
