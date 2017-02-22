import sys,argparse
import os
import cv2
from moviepy.editor import VideoFileClip
import numpy as np
import sys

from VehicleDetection import VehicleDetection

def process_video(infile,outfile, camera, vehicle):
    print("Reading {}".format(os.path.basename(infile)))

    vehicle_detection = VehicleDetection(from_pickle=vehicle)

    clip = VideoFileClip(infile)
    # Vehicle Detection
    adj_clip = clip.fl_image(vehicle_detection.process_image)

    adj_clip.write_videofile(outfile, audio=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vehicle Detection and Lane Finding')
    parser.add_argument('-i','--input', help='Input video file',required=True,dest="input")
    parser.add_argument('-o','--output',help='Output video file', required=True,dest="output")
    parser.add_argument('-c', dest="camera",help='Calibration File from calibrate.py')
    parser.add_argument('-v', dest="vehicle",help='Vehicle Detection Pickle File')
    args = parser.parse_args()

    process_video(args.input, args.output, args.camera, args.vehicle)
