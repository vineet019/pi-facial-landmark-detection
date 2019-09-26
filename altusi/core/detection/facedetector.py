"""
FaceDetector class
------------------

Class for Face detection using DNN from OpenCV
"""

import numpy as np
import cv2 as cv
import dlib

from altusi.configs import config as cfg
from altusi.utils import imgproc

from altusi.utils.logger import *

class FaceDetector:
    """Class for Face detection using DNN from OpenCV and Dlib"""

    def __init__(self, lib='dnn'):
        """Initialization for Face Detector

        Initialize DNN network given proto and model file
        provided by OpenCV official site

        Args:
        -----
            lib : str
                library for Face detection
                    * dnn: for OpenCV DNN Face detection
                    * dnn-ncs: OpenCV DNN with the support of NCS
                    * dlib: for Dlib Frontal Face detection
        """
        self.__lib = lib


        if self.__lib == 'dlib':
            self.__detector = dlib.get_frontal_face_detector()
        else:
            self.__detector = cv.dnn.readNetFromCaffe(
                cfg.CV_DNN_FACE_PROTO, cfg.CV_DNN_FACE_MODEL)
            if self.__lib.endswith('ncs'):
                self.__detector.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
                self.__detector.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
            else:
                self.__detector.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
                self.__detector.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


    def __detectFaces_dnn(self, img, default_conf=0.8):
        """Private function for detecting human faces from an image

        Inference DNN network to detect faces from the given
        input image

        Args:
        -----
            img : numpy.array
                input image
            default_conf : float
                default confidence level for Face detection
       
        Returns:
        --------
            confs : list(float)
                list of corresponding detection confidences
            bboxes : list(numpy.array(x, y, w, h) )
                list of detected faces from the input image
        """
        H, W = img.shape[:2]
        resized_img = cv.resize(img, (300, 300), interpolation=cv.INTER_CUBIC)
        blob = cv.dnn.blobFromImage(resized_img, 
                                    1., (300, 300),
                                    (104., 177., 123.) )
        self.__detector.setInput(blob)
        preds = self.__detector.forward()
        preds = np.reshape(preds, preds.shape[2:] )

        confs, bboxes = [], []
        for i, pred in enumerate(preds):
            conf = pred[2]
            if conf < default_conf: continue

            max_v = np.max(pred[3:7] )
            min_v = np.min(pred[3:7] )
            if max_v > 1 or min_v < 0: continue 

            bbox = pred[3:7] * np.array([W, H, W, H] )
            x1, y1, x2, y2 = bbox.astype('int')
            bboxes.append((x1, y1, x2-x1, y2-y1) )
            confs.append(conf)
        return confs, bboxes


    def __detectFaces_dlib(self, img):
        """Private function for detecting human faces from an image

        Apply Dlib Frontal face detector

        Args:
        -----
            img : numpy.array
                input image
        
        Returns:
        --------
            bboxes : list(numpy.array(x, y, w, h) )
                list of detected faces from the input image
        """
        rectangles = self.__detector(img)
        return [imgproc.rectangle2Rect(rectangle) for rectangle in rectangles]


    def getFaces(self, img, default_conf=0.8):
        """Detect human faces from an input image

        Args:
        -----
            img : numpy.array
                input image
            default_conf : float
                default confidence level for Face detection
                (only applied when library is OpenCV-DNN)
        
        Returns:
        --------
            confs : list(float)
                list of corresponding detection confidences
                (not available is using `Dlib` - return None)
            bboxes : list(numpy.array(x, y, w, h) )
                list of detected faces from the input image
        """
        if self.__lib == 'dlib':
            return None, self.__detectFaces_dlib(img)
        else:
            return self.__detectFaces_dnn(img, default_conf)
