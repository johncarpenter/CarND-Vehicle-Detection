import sys,argparse
import os
import cv2
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import math
import matplotlib.gridspec as gridspec
import random

from Config import Config
from VehicleDetection import VehicleDetection
'''
This program is designed to take in a series of images classified as either
cars or not-cars and create a LinearSVC classifier from it. Outputs a pickle
file (vehicle_detection.p) that is used in the image and video processors
'''
def train(cars,notcars):
    '''
    Runs the training algorithm on a series of images for the LinearSVC classifier
    Parameters:
    cars Listing of images as files as png or jpg for processing
    notcars Listing of images as files for the "not cars" parameterization
    '''
    print("Training...");
    vehicle_detection = VehicleDetection(config=Config())
    vehicle_detection.train(cars,notcars)
    print("Training finished")

    # Optional visualization
    '''
    car_files = glob.glob(notcars+'/**/*.png', recursive=True)
    test_image = random.choice(car_files)
    img = mpimg.imread(test_image)
    feature,vis = vehicle_detection.single_image_features(img,debug=True)
    visualize_training(vis)
    '''

    #test_sample(vehicle_detection)


def visualize_training(training_images):

    images = []
    images.append((training_images['raw'],"Non Car"))
    images.append((training_images['color_space'],"HLS Conversion",'gist_rainbow_r'))

    images.append((training_images['hog_0'],"HOG 0"))
    images.append((training_images['hog_1'],"HOG 1"))
    images.append((training_images['hog_2'],"HOG 2"))
    render_results(images,output="../output_images/training_non_car_sample.jpg")

def test_sample(vehicle_detection):
    '''
    Takes the model and runs the detection on 6 test images.

    Parameters:
    vehicle_detection VehicleDetection class

    Output Plot with the test images rendered
    '''
    test_images = glob.glob('../test_images/*.jpg')
    images = []
    for img_name in test_images:
        raw_image = mpimg.imread(img_name)
        out_img = vehicle_detection.process_image(raw_image)

        # Because sequential images create a historical helpers, we have to
        # delete the historical records after each image
        vehicle_detection.prev_hot_windows = []
        vehicle_detection.tracked_cars = []

        images.append((out_img,img_name))

    render_results(images)


def render_results(images, images_per_row = 2,output = False):

        nrow = math.ceil(len(images) / images_per_row)

        gs = gridspec.GridSpec(nrow,images_per_row)

        fig = plt.figure()
        fig.set_size_inches(8,12)
        for ndx,pair in enumerate(images):
            ax = fig.add_subplot(gs[ndx])
            ax.set_title("{}".format(pair[1]))
            cmap = 'gray' if len(pair) == 2 else pair[2]
            print(cmap)
            ax.imshow(pair[0],cmap=cmap)

        if(output):
            plt.savefig(output, dpi=300)
        else:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vehicle Detection Training Program')
    parser.add_argument('cars', help='Cars image directory')
    parser.add_argument('notcars', help='Not cars image directory')
    #todo pass in config text file

    args = parser.parse_args()

    '''
    All output is redirected to training.log file. @TODO parameterize this
    '''
    old_stdout = sys.stdout

    log_file = open("training.log","w")

    sys.stdout = log_file

    train(args.cars,args.notcars)

    # Close the log file
    sys.stdout = old_stdout

    log_file.close()
