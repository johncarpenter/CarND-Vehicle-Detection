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

from Config import Config
from VehicleDetection import VehicleDetection

def train(cars,notcars):
    print("Training...");
    vehicle_detection = VehicleDetection(config=Config())
    vehicle_detection.train(cars,notcars)
    print("Training finished")
    test_sample(vehicle_detection)

def test_sample(vehicle_detection):
    test_images = glob.glob('../test_images/*.jpg')
    images = []
    for img_name in test_images:
        raw_image = mpimg.imread(img_name)
        out_img = vehicle_detection.process_image(raw_image)
        images.append((out_img,img_name,'brg'))

    render_results(images)


def render_results(images, images_per_row = 3,output = False):

        nrow = math.ceil(len(images) / images_per_row)

        gs = gridspec.GridSpec(nrow,images_per_row)

        fig = plt.figure()
        fig.set_size_inches(8,12)
        for ndx,pair in enumerate(images):
            ax = fig.add_subplot(gs[ndx])
            ax.set_title("{}".format(pair[1]))
            ax.imshow(pair[0],cmap=pair[2])

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

    #redirect logs to file
    old_stdout = sys.stdout

    log_file = open("training.log","w")

    sys.stdout = log_file

    train(args.cars,args.notcars)

    # Close the log file
    sys.stdout = old_stdout

    log_file.close()
