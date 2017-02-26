import sys,argparse
import os
import cv2
from moviepy.editor import VideoFileClip
import numpy as np
import sys

from VehicleDetection import VehicleDetection
from LaneDetector import LaneDetector

class MultipleDetector:
    '''
    Class to handle lane and vehicle detector processors.

    @TODO should be expanded to be more generic and detectors classes abstracted
    '''
    def __init__(self, lane_detector=None, vehicle_detection=None):
        '''
        Parameters:
        lane_detector Detector class
        vehicle_detection VehicleDetection class
        '''
        self.lane_detector = lane_detector
        self.vehicle_detector = vehicle_detection
        self.frame_count = 0

    def process_image(self,img):
        '''
        Process individual image or image from video stream

        Parameters:
        img PIL format image
        '''
        self.frame_count += 1


        if self.lane_detector:
            img = self.lane_detector.process_image(img)

        if self.vehicle_detector:
            img = self.vehicle_detector.process_image(img, full_search=self.frame_count % 1 == 0)
        return img

def process_video(infile,outfile, camera=None, vehicle=None):
    '''
    Load video, process individual image frames, output frames to new Video
    Parameters:
    infile Filename of input video
    outfile Filename of output video
    camera Filename of Camera Calibration pickle (from calibration.py)
    vehicle Filename of Vehicle Model pickle (from train.py)
    '''
    print("Reading {}".format(os.path.basename(infile)))

    vehicle_detection = VehicleDetection(from_pickle=vehicle)

    lane_detector = LaneDetector(use_smoothing = True, camera = camera)

    multiple_detector = MultipleDetector(vehicle_detection=vehicle_detection)

    clip = VideoFileClip(infile)

    adj_clip = clip.fl_image(multiple_detector.process_image)
    adj_clip.write_videofile(outfile, audio=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vehicle Detection and Lane Finding')
    parser.add_argument('-i','--input', help='Input video file',required=True,dest="input")
    parser.add_argument('-o','--output',help='Output video file', required=True,dest="output")
    parser.add_argument('-c', dest="camera",help='Calibration File from calibrate.py')
    parser.add_argument('-v', dest="vehicle",help='Vehicle Detection Pickle File')
    args = parser.parse_args()

    process_video(args.input, args.output, args.camera, args.vehicle)
