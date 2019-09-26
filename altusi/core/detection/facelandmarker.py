"""
FaceLandmarker class
====================

Class for Face landmark detection
"""

import dlib

from altusi.configs import config as cfg
from altusi.utils import imgproc
from altusi.utils.logger import *

class FaceLandmarker:
    def __init__(self, model_path=cfg.DLIB_FACIAL_LANDMARK_MODEL):
        """Initialization for Face Landmarker
        
        Initialize predictor for locating facial landmark

        Arguments:
        ----------
            model_path : str
                path to landmark predictor model
        """

        self.__predictor = dlib.shape_predictor(model_path)


    def findLandmark(self, image, bbox):
        """Locate facial landmark from a detected face in an image
        
        Given an image and a bounding box of a detected face, 
        return a list of facial points corresponding to the input face.
        Each point is represented by a tuple.

        Arguments:
        ----------
            image : numpy.array
                input colored image for locating facial landmark
            bbox : np.array([x, y, w, h] )
                face's bounding box

        Returns:
        --------
            landmark : list(tuple)
                List of facial points' coordinates
        """

        rect = imgproc.rect2Rectangle(bbox)
        shape = self.__predictor(image, rect)
        landmark = imgproc.shape2Points(shape)
        return landmark


    def findLandmarks(self, image, bboxes):
        """Locate facial landmarks from detected faces in an image

        Given an image and a bounding box of detected faces, 
        return a list of facial points corresponding to each input face.
        Each point is represented by a tuple.

        Arguments:
        ----------
            image : numpy.array
                input colored image for locating facial landmarks
            faces : list(np.array([x, y, w, h] ) )
                list of faces' bounding boxes

        Returns:
        --------
            landmarks: list(list(tuple) )
                list of List of facial points' coordinates
        """

        landmarks = []
        for bbox in bboxes:
            landmark = self.findLandmark(image, bbox)
            landmarks.append(landmark)
        return landmarks
