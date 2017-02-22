import sys,argparse
import os
import cv2
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

from Config import Config
from VehicleDetection import VehicleDetection

def train(cars,notcars):
    print("Training...");
    vehicle_detection = VehicleDetection(config=Config())
    vehicle_detection.train(cars,notcars)
    print("Training finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vehicle Detection Training Program')
    parser.add_argument('cars', help='Cars image directory')
    parser.add_argument('notcars', help='Not cars image directory')
    #todo pass in config text file

    args = parser.parse_args()
    train(args.cars,args.notcars)
