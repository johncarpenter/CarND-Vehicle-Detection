from utils import *
import numpy as np
import cv2

class CarBlob:
    def __init__(self, bbox):
        '''
        Class to hold predicted vehicle parameters within an image

        @TODO expand class to handle vehicle movement and predicted motion
        '''
        self.centroid = None
        self.bbox = []
        self.update_flag = False
        self.update(bbox)

    def update(self, bbox):
        '''
        Updates the vehicle parameters within a frame. Also sets the update_flag
        which can be used to remove CarBlob objects which have disappeared

        Parameters:
        bbox Pair of tuples holding the image coordinates of the estimated position ((top left),(bottom right))
        '''
        if(self.bbox):
            delta = np.subtract(bbox,self.bbox)
            change = [0.1*x for x in delta]
            change = tuple(map(tuple,change))
            new_bbox_tl = tuple(int(sum(x)) for x in zip(self.bbox[0],change[0]))
            new_bbox_br = tuple(int(sum(x)) for x in zip(self.bbox[1],change[1]))
            self.bbox = (new_bbox_tl,new_bbox_br)

        else:
            self.bbox = bbox

        self.centroid = (int((bbox[0][0]+bbox[1][0])/2), int((bbox[0][1]+bbox[1][1])/2))
        self.update_flag = True

    def overlaps(self,check_bbox):
      '''
      Calculates if the check_bbox intersects with the current bbox

      Parameters:
      bbox Pair of tuples holding the image coordinates of the estimated position ((top left),(bottom right))
      Returns:
      True if the boxes intersect
      '''
      if not self.bbox:
          return False

      dx = min(self.bbox[1][0], check_bbox[1][0]) - max(self.bbox[0][0],check_bbox[0][0])
      dy = min(self.bbox[1][1], check_bbox[1][1]) - max(self.bbox[0][1], check_bbox[0][1])
      if (dx>=0) and (dy>=0):
          return True

      return False

    def draw(self,img):
        '''
        Overlays the bbox onto an image
        '''
        #cv2.circle(img, self.centroid, 3, (0,255,0), thickness=4)
        cv2.rectangle(img,self.bbox[0],self.bbox[1],(0,255,0),thickness=4)


    def dump(self):
        '''
        Debug output
        '''
        print("Centroid {}".format(self.centroid))
        print("Bounding {}".format(self.bbox))
