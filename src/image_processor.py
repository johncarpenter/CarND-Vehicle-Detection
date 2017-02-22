import sys,argparse
import os
import cv2
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt


from VehicleDetection import VehicleDetection

def process_image(infile,outfile,vehicle):
    print("Processing {}".format(infile))
    vehicle_detection = VehicleDetection(from_pickle=vehicle)

    raw_image = mpimg.imread(infile)

    out_img = vehicle_detection.process_image(raw_image)

    plt.imshow(out_img,cmap='brg')
    if(outfile):
        plt.savefig(outfile, dpi=300)
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
