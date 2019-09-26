"""
Drawer library
==============

Library to ease the drawing features.
Its functions are supported by OpenCV and Pillow

Revision
--------
    2019, Apr 13:
        - Add functions to draw circles `drawCircle` and `drawCircles`
    2019, Apr 11:
        - Add function to draw info `drawInfo`
"""

import numpy as np
from PIL import Image, ImageFont, ImageDraw

from altusi.configs import config as cfg
from altusi.utils.logger import *

# colors for drawing
COLOR_RED = (0, 0, 255)
COLOR_RED_LIGHT = (49, 81, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_LIGHT_SKY_BLUE = (250, 206, 135)
COLOR_DEEP_SKY_BLUE = (255, 191, 0)
COLOR_PURPLE = (255, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


#===============================================================================
# DRAWING FUNCTIONS
#===============================================================================

def drawCircle(image, center, radius=1, color=COLOR_YELLOW, thickness=-1):
    """Draw a circle on an input image
    
    Arguments:
    ----------
        image : numpy.array
            input image for drawing
        center : tuple(int, int)
            center's coordinate of the drawing circle 
    
    Keyword Arguments:
    ------------------
        radius : int (default: 1)
            radius of the circle 
        color : tuple(B : int, G : int, R: int) (default: COLOR_YELLOW)
            drawing color for the circle
        thickness : int (default: -1)
            how thick the circle is
            negative thickness means a filled circle is to be drawn
    """

    cv.circle(image, center, radius, color, thickness)


def drawCircles(image, centers, radius=1, color=COLOR_YELLOW, thickness=-1):
    """Draw circles on an input image
    
    Arguments:
    ----------
        image : numpy.array
            input image for drawing
        centers : tuple(int, int)
            centers' coordinate of the drawing circles
    
    Keyword Arguments:
    ------------------
        radius : int (default: 1)
            radius of the circles
        color : tuple(B : int, G : int, R: int) (default: COLOR_YELLOW)
            drawing color for the circle
        thickness : int (default: -1)
            how thick the circle is
            negative thickness means a filled circle is to be drawn
    """

    for i, center in enumerate(centers):
        drawCircle(image, center, radius, color, thickness)


def drawObjects(image, objects, color=COLOR_YELLOW, thickness=2):
    """Draw bounding boxes for given input objects

    Arguments:
    ----------
        image : numpy.array
            input image for drawing
        objects : list(numpy.array([x, y, w, h] ) )
            input bounding boxes of objects to draw

    Keyword Arguments:
    ------------------
        color : tuple(B : int, G : int, R: int) (default: COLOR_YELLOW)
            drawing color for drawing 
        thickness : int (default: 2)
            how thick the shape is

    Returns:
    --------
        image : numpy.array
            output image after drawing
    """

    if len(objects) == 0:
        return image
        exit()

    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for (x, y, w, h) in objects:
        for i in range(thickness):
            draw.rectangle([x+i, y+i, x+w-i, y+h-i], outline=color)
        
    del draw
    return np.asarray(image)


def drawLabels(image, objects, labels, color=COLOR_RED, thickness=2):
    """Draw bounding boxes and the corresponding labels for given input objects

    Arguments:
    ----------
        image : numpy.array
            input image for drawing
        objects : list(numpy.array([x, y, w, h] ) )
            input bounding boxes of objects to draw
        labels  : list(list(str) )
            corresponding labels for input objects

    Keyword Arguments:
    ------------------
        color : tuple(B : int, G : int, R: int) (default: COLOR_RED)
            drawing color for drawing 
        thickness : int (default: 2)
            how thick the shape is

    Returns:
    --------
        image : numpy.array
            output image after drawing
    """

    if len(objects) == 0:
        return image
        exit()

    font = ImageFont.truetype(font=cfg.FONT, \
                    size=np.floor(3e-2 * image.shape[0] + 0.5).astype('int32') )
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for (x, y, w, h), label in zip(objects, labels):
        # label_size = np.array([w, (draw.textsize(label, font) )[1] ] )
        label = '{}'.format(label)
        label_size = draw.textsize(label, font) 

        if y - label_size[1] >= 0:
            text_coor = np.array([x, y - label_size[1] ] )
        else:
            text_coor = np.array([x, y + 1] )

        for i in range(thickness):
            draw.rectangle([x+i, y+i, x+w-i, y+h-i], outline=color)
        
        draw.rectangle([tuple(text_coor), tuple(text_coor + label_size) ], fill=color)
        draw.text(text_coor, label, fill=COLOR_WHITE, font=font)
    del draw
    return np.asarray(image)


def drawInfo(image, labels, color=COLOR_RED):
    """Draw information label for an image
    
    Arguments:
    ----------
        image : numpy.array
            input image for drawing
        labels : list(str)
            list of label to draw
    
    Keyword Arguments:
    ------------------
        color : tuple(B : int, G : int, R: int) (default: COLOR_RED)
            color for drawing
    
    Returns:
    --------
        image : numpy.array
            output image after drawing
    """

    font = ImageFont.truetype(font=cfg.FONT, \
                size=np.floor(3e-2 * image.shape[0] + 0.5).astype('int32') )
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    prv_y = 0
    for i, label in enumerate(labels):
        label_size = list(draw.textsize(label, font) )
        label_size[0] += 1 
        label_size[1] += 1 

        text_coor = np.array([0, prv_y] )
        prv_y += label_size[1] + 1
        label_size = tuple(label_size)

        draw.rectangle([tuple(text_coor), tuple(text_coor + label_size) ], fill=color)
        draw.text(text_coor, label, fill=COLOR_WHITE, font=font)

    del draw
    return np.asarray(image)
