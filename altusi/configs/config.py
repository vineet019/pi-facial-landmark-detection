import os

#===============================================================================
# PROJECT'S ORGANIZATION
#===============================================================================
PROJECT_BASE = '.'


#===============================================================================
# PROJECT'S PARAMETERS
#===============================================================================
FONT = os.path.join(PROJECT_BASE, 'altusi', 'utils', 'FiraMono-Medium.otf')


#===============================================================================
# PROJECT'S MODELS
#===============================================================================
MODEL_DIR = 'models'

HUMAN_FACE_MODEL_DIR = os.path.join(MODEL_DIR, 'human-face')

# Face detection
CV_DNN_FACE_PROTO = os.path.join(HUMAN_FACE_MODEL_DIR, 'deploy.prototxt.txt')
CV_DNN_FACE_MODEL = os.path.join(HUMAN_FACE_MODEL_DIR, 
                                'res10_300x300_ssd_iter_140000.caffemodel')

# Facial landmark detection
DLIB_FACIAL_LANDMARK_MODEL = os.path.join(HUMAN_FACE_MODEL_DIR,
                                    'shape_predictor_68_face_landmarks.dat')


#===============================================================================
# IMAGE'S PARAMETERS
#===============================================================================


#===============================================================================
# DEBUG MODE 
#===============================================================================
