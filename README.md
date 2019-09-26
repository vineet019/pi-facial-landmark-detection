# Raspberry Pi - Facial Landmark Detection

by Danh Doan


## Introduction
This project aims to perform Facial Landmark Detection on Raspberry Pi board. 
In Face Detection step, Dlib detector [[link]](http://dlib.net/imaging.html#get_frontal_face_detector), OpenCV-DNN detector [[link]](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) are used to draw comparison between processing speed and accuracy. For OpenCV-DNN detector, it is also accelerated by the support of Movidius Neural Compute Stick [[link]](https://software.intel.com/en-us/movidius-ncs).
After faces are detected, shape predictor supported [[link]](http://dlib.net/face_landmark_detection.py.html) from Dlib is utilized to locate facial landmarks.

## Demonstration Video
Youtube: [[link]](https://youtu.be/WzvgrhrDC1s)

## Usage
**1. Clone this repository**
`git clone https://github.com/danhdoan/pi-facial-landmark-detection`

This is a plug-n-play project, all source code and models are available after downloaded

**2. Perform the demonstration**
**2.0 Argument parser:**

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
	
**2.1 Apply Dlib Face detector:**
	`python3 landmark-detector.py --lib dlib`

In the demo video, a Webcam is used so I use `-fh` argument to flip frames horizontally
	
**2.2 Apply OpenCV-DNN Face detector:**
	`python3 landmark-detector.py --lib dnn`
	
**2.2 Apply OpenCV-DNN Face detector with NCS support:**
	`python3 landmark-detector.py --lib dnn-ncs`

## Performance Comparision

| Detector   | Backend |  FPS |
|-------------|-----|-- |
| Dlib | Raspi CPU |~0.96|
| OpenCV-DNN| Raspi CPU| ~1.17 |
|| Movidius NCS | ~5.45

