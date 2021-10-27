import sys
import cv2
import argparse
import os
import logging
from line_profiler import LineProfiler
from imutils.video import WebcamVideoStream  # threaded version
import imutils
from imutils.video import FPS
import logging
import traceback

# set logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def frame_save_as_jpg(frame, fps):
    # frame = imutils.resize(frame, width=400)
    directory = os.path.join(os.path.dirname(__file__), './jpgs')
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(f"jpgs/{fps._numFrames}.jpg", frame)


MAX_FRAME_NUM = float("inf")
def readfrom_webcam():
    cap = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    fps.stop()

    # check camera is opened or not.
    if cap.stream is None or not cap.stream.isOpened():
        print('\n\n')
        print('Error - could not open video device.')
        print('\n\n')
        exit(0)

    while True and fps._numFrames < MAX_FRAME_NUM:

        # frame = cap.read()
        # logger.info(f"grab: {cap.stream.grab()}")
        if cap.stream.grab():
            # logger.info(f"retrieve: {cap.stream.retrieve()}")
            has_frame, frame = cap.stream.retrieve() # retrieve returns tuple type
            if not has_frame:
                logger.info(f"frame is empty")
                continue

            logger.info(f"type(frame): {type(frame)}")
            frame_save_as_jpg(frame, fps)

            # FPS info
            logger.info("approx. FPS/elasped_time/#frames: {:.2f}/{:.2f}/{}".format(fps.fps(), fps.elapsed(), fps._numFrames))

        # update the FPS counter
        fps.update()

        # stop() method updates the _end attribute
        fps.stop()

        # check if the key
        # key = cv2.waitKey(1)
        # if key < 0:
        #     logger.info(f"key<0 : {key}")
        #     break

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()

def readfrom_videofile(inputfile):
    inputfile = inputfile
    cap = cv2.VideoCapture(inputfile)
    fps = FPS().start()
    fps.stop()
    while cv2.waitKey(1) < 0 and fps._numFrames < MAX_FRAME_NUM:
        has_frame, frame = cap.read()
        fps.update()
        fps.stop()
        if not has_frame:
            logger.info("no frame!")
            break
        cv2.imwrite(f"./jpgs/{fps._numFrames}.jpg", frame)
        window_name = 'my input video'
        cv2.imshow(window_name, frame)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        logger.info(f"_numFrames: {fps._numFrames}")
    cv2.destroyWindow()

try:
    readfrom_webcam()
    # readfrom_videofile("./test.mov")
except Exception as e:
    logger.exception(e)
    logger.debug(traceback.format_exc())
    traceback.print_exc()