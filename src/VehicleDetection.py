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
from CarBlob import *

class VehicleDetection:
    '''
    Initialize the VehicleDetection object. Must be trained or provided with a
    pickle file from previous training

    Parameters:
    config: Config object.
    from_pickle: Filename (vehicle_detection.p) of detection model
    '''
    def __init__(self,config = None, from_pickle=None):
        self.config = config
        self.svc = None
        self.xscaler = None
        self.has_trained = False
        if(from_pickle):
            self.__load_from_pickle(filename=from_pickle)

        '''Container of CarBlob objects that maintain a running list of cars in the scene
        '''
        self.tracked_cars = []

        '''Stack of the previous calculated hot windows. Used to improved detection per scene
        '''
        self.prev_hot_windows = []

        '''Test Background filter. Not used @TODO determine if this can be used accuractely to remove
        background from cars
        '''
        self.background_filter = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=5)

    def __load_from_pickle(self,filename="vehicle_detection.p"):
        '''
        Loads the current model and configuration from pickle file.
        '''
        #TODO handle missing file or errors
        dist_pickle = pickle.load( open( filename, "rb" ) )
        self.svc = dist_pickle["svc"]
        self.xscaler = dist_pickle["xscaler"]
        self.config = dist_pickle["config"]
        self.has_trained = True


    def train(self, car_image_directory, not_car_image_directory):
        '''
        Trains the LinearSVC model and outputs the vehicle_detection.p file.
        Currently only loads png files

        Parameters:
        car_image_directory: Filename listing of images representing cars
        not_car_image_directory: Filename listing of images representing "not cars"
        '''
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

    def process_image(self,img, full_search=True):
        '''
        Applies the Vehicle Detection algorithms to a single image. When images
        are passed in sequentially the historical filters are used.

        @todo disable the historical filters on demand
        '''
        if(self.has_trained == False):
            sys.exit("Config or training file not provided. Exiting")

        if(full_search):
            # Define the full search area
            x_start_stop = [None,None]
            x_start_stop[0] = 400
            x_start_stop[1] = img.shape[1]

            y_start_stop = [None,None]
            y_start_stop[0] = int(img.shape[0]/2)
            y_start_stop[1] = img.shape[0]-80

            # Separate out background images
            #fgmask = self.background_filter.apply(img)
            #img = cv2.bitwise_and(img,img,mask=fgmask)

            hot_windows = []

            hot_windows += self.__search(img,SearchConfig(x_start_stop=x_start_stop,y_start_stop=y_start_stop, xy_window=(52,52),xy_overlap=(0.5,0.5)))
            hot_windows += self.__search(img,SearchConfig(x_start_stop=x_start_stop,y_start_stop=y_start_stop, xy_window=(110,110),xy_overlap=(0.5,0.5)))

            # Keep a running record for the last 500ms
            self.prev_hot_windows.append(hot_windows);
            if(len(self.prev_hot_windows) >= 12):
                self.prev_hot_windows.pop(0)

            # Append all the information from the last 12 frames into one estimator
            hot_windows = []
            for hw in self.prev_hot_windows:
                hot_windows += hw

            # Debug - Draw search windows
            window_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=2)

            #Create heatmap style image from overlaid windows.
            #thresh = int(len(self.prev_hot_windows)*0.75)  #Threshold is based upon size of historical data
            heatmap = filter_by_heatmap(img, hot_windows,thresh=12)

            labels = label(heatmap)

            # Itearate through labels and identify vehicles
            self.__identify_vehicles(labels)

            #Debug overlay the search windows
            img = cv2.addWeighted(img, 1, window_img, 0.8, 0)

        # Draw each of the tracked vehicles
        for car in self.tracked_cars:
            car.draw(img)

        # Debug - Draw the raw label output
        #draw_img = draw_labeled_bboxes(raw_image, labels)




        return img

    def __identify_vehicles(self, labels):
        '''
        Iterates through the labels to find boxes. Compares the boxes to
        tracked_cars and either updates or adds a new tracked car. Finally, removes
        any car that wasn't updated in the last iteration.

        Parameters:
        labels: Output from scipy.ndimage.measurements.label
        '''
        bboxes = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            bboxes.append(bbox)

        for bbox in bboxes:
            found_match = False
            for car in self.tracked_cars:
                if car.overlaps(bbox):
            #        print("Updating")
                    car.update(bbox)
                    found_match = True
                    break

            if not found_match:
            #    print("Adding Car")
                self.tracked_cars.append(CarBlob(bbox))

        for car in self.tracked_cars:
            if not car.update_flag:
            #    print("Removing Car")
                self.tracked_cars.remove(car)
            car.update_flag = False

        #print("Number of Tracked Cars {}".format(len(self.tracked_cars)))


    def __search(self,img, search_config):
        '''
        Searches the image using the provided SearchConfig
        Parameters:
        img: PIL image
        search_config: SearchConfig object with the parameters for the sliding windows
        '''
        #print(search_config.dump())
        windows = slide_window(img, y_start_stop=search_config.y_start_stop, x_start_stop=search_config.x_start_stop,
                            xy_window=search_config.xy_window,xy_overlap=search_config.xy_overlap)
        return self.__search_windows(img, windows)


    def single_image_features(self,img,debug=False):
        '''
        Creates a feature list of for an image. Based on the configuration provided
        Parameters:
        debug: Outputs a series of images for visualization

        Output:
        List of features

        '''

        output_vis = {}
        if(debug):
            output_vis['raw'] = img.copy()


        #1) Define an empty list to receive features
        img_features = []
        #2) Apply color conversion if other than 'RGB'
        feature_image = convert_image(img,color_space=self.config.color_space)

        if(debug):
            output_vis['color_space'] = feature_image.copy()

        #3) Compute spatial features if flag is set
        if self.config.spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=self.config.spatial_size)
            #4) Append features to list
            img_features.append(spatial_features)

            if(debug):
                output_vis['spatial_features'] = spatial_features

        #5) Compute histogram features if flag is set
        if self.config.hist_feat == True:
            hist_features = color_hist(feature_image, nbins=self.config.hist_bins, bins_range=(0,256))
            #6) Append features to list
            img_features.append(hist_features)

            if(debug):
                output_vis['hist_features'] = hist_features

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

            if(debug):
                hog,image = get_hog_features(feature_image[:,:,0], self.config.orient,
                            self.config.pix_per_cell, self.config.cell_per_block, vis=True, feature_vec=True)
                output_vis['hog_0'] = image
                hog,image = get_hog_features(feature_image[:,:,1], self.config.orient,
                            self.config.pix_per_cell, self.config.cell_per_block, vis=True, feature_vec=True)
                output_vis['hog_1'] = image
                hog,image = get_hog_features(feature_image[:,:,2], self.config.orient,
                            self.config.pix_per_cell, self.config.cell_per_block, vis=True, feature_vec=True)
                output_vis['hog_2'] = image

            #8) Append features to list

        #9) Return concatenated array of features
        if(debug):
            return np.concatenate(img_features), output_vis
        else:
            return np.concatenate(img_features)


    def __search_windows(self,img, windows):
        '''
        Searches through windows apply the LinearSVC to each window for the classifier

        Parameters:
        img PIL image
        windows: List of window bounding boxes ((top left),(bottom right))
        '''
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            #4) Extract features for that window using single_img_features()
            features = self.single_image_features(test_img)
            #5) Scale extracted features to be fed to classifier
            test_features = self.xscaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = self.svc.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows
