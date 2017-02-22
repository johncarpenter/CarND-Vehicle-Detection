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

from VehicleDetection import VehicleDetection

def process_image(infile,outfile,vehicle):
    print("Processing {}".format(infile))
    vehicle_detection = VehicleDetection(from_pickle=vehicle)
    '''
    raw_image = mpimg.imread(infile)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    raw_image = raw_image.astype(np.float32)/255



    out_img = vehicle_detection.process_image(raw_image)


    plt.imshow(out_img,cmap='brg')
    if(outfile):
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()
    '''
    test_sample(vehicle_detection)

def test_sample(vehicle_detection):
    test_images = glob.glob('../test_images/*.jpg')
    images = []
    for img_name in test_images:
        raw_image = mpimg.imread(img_name)
        raw_image = raw_image.astype(np.float32)/255
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
    parser = argparse.ArgumentParser(description='Image Preprocessing Testing Tool')
    parser.add_argument('input', type=argparse.FileType('r'),help='test images file')
    parser.add_argument('-c', dest="camera",help='Calibration File from calibrate.py')
    parser.add_argument('-v', dest="vehicle",help='Vehicle Detection Pickle File')
    parser.add_argument('-o','--output',help='Output test file', dest="output")


    args = parser.parse_args()

    output_name = args.output.name if args.output != None else None

    process_image(args.input.name, output_name, args.vehicle)
