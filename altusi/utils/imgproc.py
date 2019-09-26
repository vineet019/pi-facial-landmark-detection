"""
Imgproc library
===============

Library to support Image processing functions

Revision
--------
    2019, Apr 13:
        - Change returned datatype of `shape2Points`
"""

import os
import math
import numpy as np
import cv2 as cv
import dlib

#===============================================================================
# SUPPORT FUNCTIONS
#===============================================================================

def resizeByHeight(image, height=720):
    """Resize an image given the expected height and keep the original ratio

    Arguments:
    ----------
        image : numpy.array
            input image to resize

    Keyword Arguments:
    ------------------
        height : int (default: 720)
            expected width of output image

    Returns:
    --------
        out_image : numpy.array
            output resized image
    """
    H, W = image.shape[:2]
    width = int(1. * W * height / H + 0.5)
    out_image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)

    return out_image


def resizeByWidth(image, width=600):
    """Resize an image given the expected width and keep the original ratio

    Arguments:
    ----------
        image : numpy.array
            input image to resize

    Keyword Arguments:
    ------------------
        width : int (default: 600)
            expected width of output image

    Returns:
    --------
        out_image : numpy.array
            output colored image after resized
    """

    H, W = image.shape[:2]
    height = int(H * width / W)
    out_image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)
    return out_image


def cameraCalibrate(capturer, size=None, by_height=False):
    """Get camera's information like dimension and FPS

    Arguments:
    ----------
        capturer : cv.VideoCapture
            OpenCV-Video capturer object

    Keyword Arguments:
    ------------------
        width : int (default: None)
            width value to resize by width

    Returns:
    --------
        (W, H) : int, int
            dimension of video's frame
        FPS : float
            FPS of the video stream
    """

    fps = capturer.get(cv.CAP_PROP_FPS)

    while True:
        _, frame = capturer.read()
        if _:
            if size:
                if by_height:
                    frame = resizeByHeight(frame, size)
                else:
                    frame = resizeByWidth(frame, size)
            H, W = frame.shape[:2]

            return (W, H), fps


def prewhiten(image):
    """Preprocess an image by zero-mean and unit-variance transforms

    Arguments:
    ----------
        image : numpy.array
            input colored image

    Returns:
    --------
        out_image : numpy.array
            output image after preprocessed
    """

    mean = np.mean(image)
    std = np.std(image)
    std_adj = np.maximum(std, 1.0/np.sqrt(image.size) )
    out_image = np.multiply(np.subtract(image, mean), 1/std_adj)
    return out_image  


def rectangle2Rect(rectangle):
    """Convert Dlib-rectangle to OpenCV-Rect datatype

    Arguments:
    ----------
        rectangle : Dlib-rectangle
            input rectangle

    Returns:
    --------
        OpenCV-Rect - np.array([x, y, w, h] )
            output corresponding OpenCV-Rect
    """

    y = rectangle.top()
    x = rectangle.left()
    h = rectangle.bottom() - y
    w = rectangle.right() - x
    return (x, y, w, h)


def rect2Rectangle(rect):
    """Convert OpenCV-Rect to Dlib-Rectangle datatype

    Arguments:
    ----------
        rect : np.array([x, y, w, h] ) 
            OpenCV-Rect type input

    Returns:
    --------
        rectangle
            the corresponding Dlib-rectangle type output
    """

    x, y, w, h = rect
    return dlib.rectangle(x, y, x+w, y+h)


def shape2Points(shape):
    """Convert Dlib-Shape datatype to a list of coordinates

    Arguments:
    ----------
        shape : Dlib-Shape
            nput shape object

    Returns:
    --------
        points : numpy.array 
            an 2D array represented coordinates
    """

    points = []
    for i in range(68):
        points.append((shape.part(i).x, shape.part(i).y) )
    return points


def getEuclideanDist(u, v):
    """Compute Euclidean distance between 2 vectors

    Arguments:
    ----------
        u : numpy.array
            first input vector
        v : numpy.array
            second input vector

    Returns:
    --------
        float
            Euclidean distance between 2 input vectors
    """

    return np.sqrt(np.sum(np.square((np.subtract(u, v) ) ) ) )
