import argparse
from altusi.utils.logger import *

def getArgs():
    """Argument collecting and parsing

    Collect and parse user arguments for program

    Returns:
    --------
        args : argparse object 
            arguments after parsing
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', type=str,
                        required=False,
                        help='path to video stream')
    parser.add_argument('--flip_hor', '-fh',
                        default=False, required=False,
                        action='store_true',
                        help='horizontally flip video')
    parser.add_argument('--flip_ver', '-fv',
                        default=False, required=False,
                        action='store_true',
                        help='vertically flip video')
    parser.add_argument('--show', '-s', 
                        default=True, required=False,
                        action='store_true',
                        help='whether or not the output is visualized')
    parser.add_argument('--name', '-n', type=str,
                        default='camera', required=False,
                        help='name of video stream used for recording')
    parser.add_argument('--lib', '-l', type=str,
                        default='dnn', required=False,
                        help='name of face detector in use')

    args = parser.parse_args()

    return args 
