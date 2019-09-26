"""
Facial Landmark Detection
==========================

Example of using Dlib to detect Facial landmarks from a video stream. 
Face detection is performed every single frame then detect facial landmarks

usage: landmark-detector.py [-h] [--video VIDEO] [--flip_hor] [--flip_ver]
                            [--show] [--name NAME] [--lib LIB]

optional arguments:
    -h, --help            show this help message and exit
    --video VIDEO, -v VIDEO
                          path to video stream
    --flip_hor, -fh       horizontally flip video
    --flip_ver, -fv       vertically flip video
    --show, -s            whether or not the output is visualized
    --name NAME, -n NAME  name of video stream used for recording
    --lib LIB, -l LIB     name of face detector in use

Keys
----
    SPACE: pause
    ESC/q: quit
"""

import numpy as np
import cv2 as cv
import dlib

from altusi.configs import config as cfg
from altusi.core.detection import FaceDetector
from altusi.core.detection import FaceLandmarker 

from altusi.helper import funcs as fn
from altusi.utils import drawer, imgproc
from altusi.utils.logger import *


def app(video_link, video_name, lib, show=True, flip_hor=False, flip_ver=False):
    # initialize Video writer
    cap = cv.VideoCapture(video_link)
    (H, W), FPS = imgproc.cameraCalibrate(cap, False)
    LOG(INFO, 'Camera info: {}\n'.format((H, W, FPS) ) )

    LOG(INFO, 'Face Detector in Use:', lib)
    face_detector = FaceDetector(lib=lib)
    face_landmarker = FaceLandmarker() 

    cnt_frm = 0
    playing = True
    while cap.isOpened():
        if playing:
            _, frm = cap.read()
            if not _:
                LOG(INFO, 'Reached the end of Video stream')
                break

            if flip_ver: frm = cv.flip(frm, 0)
            if flip_hor: frm = cv.flip(frm, 1)

            frm = imgproc.resizeByHeight(frm, 600)

            cnt_frm += 1

            # just to reduce the amount of processing
            if cnt_frm % 3 != 1: continue

            # detect faces and then detect landmarks if faces are presented
            _start_t = time.time()
            confs, bboxes = face_detector.getFaces(frm)
            if len(bboxes):
                landmarks = face_landmarker.findLandmarks(frm, bboxes)
            # calculate FPS based on the processing time for each frame
            _prx_t = time.time() - _start_t


            for i, bbox in enumerate(bboxes):
                for j, point in enumerate(landmarks[i] ):
                    x, y = point
                    cv.circle(frm, (x, y), 2, drawer.COLOR_YELLOW, -1)

            frm = drawer.drawInfo(frm, ['Raspberry Pi - {} - FPS: {:.3f}'.format(lib, 1/_prx_t)])

        cv.imshow('', frm)
        key = cv.waitKey(1)
        if key == ord(' '):
            playing = not playing
        elif key in [27, ord('q') ]:
            LOG(INFO, 'Interrupted by users')
            break

    cap.release()
    cv.destroyAllWindows()


def main(args):
    video_link = args.video if args.video else 0 
    app(video_link, args.name, args.lib, args.show, args.flip_hor, args.flip_ver)


if __name__ == '__main__':
    print(__doc__)

    LOG(INFO, 'Experiment: Facial Landmark detection on Raspberry Pi')

    args = fn.getArgs()
    main(args)

    LOG(INFO, 'Process done')
