from termcolor import colored
import numpy as np
import cv2 as cv
import time

# log mode
NONE = 0
INFO = 1
DEBUG = 2
BEGIN = 4
END = 8
ERROR = 16

log_map = {
    INFO:   'INFO',
    DEBUG: 'DEBUG',
    BEGIN: 'BEGIN',
    END:   'END',
    ERROR: 'ERROR'
}

# LOG_ENABLE = NONE
# LOG_ENABLE = INFO
# LOG_ENABLE = INFO | ERROR
LOG_ENABLE = INFO | BEGIN | END | DEBUG | ERROR

#===============================================================================
# SUPPORT FUNCTIONS
#===============================================================================

def resizeImage(image, width=600):
    '''
    Resize an image given the expected width
    and keep the original ratio

    Parameters
    ----------
    image    : numpy.array
        Input colored image to resize
    width    : int, optional
        expected width of output image

    Returns
    -------
    image    : numpy.array
        Output color image after resized
    '''
    H, W = image.shape[:2]
    height = int(H * width / W)
    image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)
    return image

def showImage(title, image):
    cv.imshow(title, image)

def waitImage():
	key = cv.waitKey(0)
	if key == 27 or key == ord('q'):
		pass

# extract file name from a given file path
def getFileName(file_dir):
    start_idx = file_dir.rfind('/')
    end_idx = file_dir.rfind('.')
    if end_idx == -1:
        end_idx = len(file_dir)

    return file_dir[start_idx+1 : end_idx]

# log function used to print log while running program
def LOG(log_mode, log_txt, obj=None):
    if LOG_ENABLE & log_mode:
        if obj is not None:
            log_str = '{} {}'.format(log_txt, obj)
        else:
            log_str = '{}'.format(log_txt)

        color = None
        if log_mode == INFO:
            color = 'blue'
        elif log_mode == ERROR:
            color = 'red'
        elif log_mode == BEGIN or log_mode == END:
            color = 'white'
        elif log_mode == DEBUG:
            color = 'yellow'

        log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime() )
        log_prefix = '[{}]'.format(log_map[log_mode] )
        print('[{}]'.format(colored(log_time, color='green') ),
              '{}'.format(colored('{:<7s}'.format(log_prefix), \
                            color=color, 
                            attrs=['bold'] ) ), \
                            log_str)
    else:
        pass

# measure run time with a given number of iterations
def testTime(func, args, iters=100000):
    start_time = time.time()

    for i in range(iters):
        func(args)
    
    end_time = time.time()
    
    return (end_time - start_time)
