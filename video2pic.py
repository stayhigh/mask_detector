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
    # cap = WebcamVideoStream(src=0).start()
    cap = cv2.VideoCapture(0)
    fps = FPS().start()
    fps.stop()

    # check camera is opened or not.
    if cap is None or not cap.isOpened():
        print('\n\n')
        print('Error - could not open video device.')
        print('\n\n')
        exit(0)

    while fps._numFrames < MAX_FRAME_NUM:
        # frame = cap.read()
        # cap.stream.setExceptionMode(True)
        print('try case')
        grabbed = cap.grab()
        logger.info(f"grabbed: {grabbed}")
        if grabbed:
            has_frame, frame = cap.retrieve() # retrieve returns tuple type
            if not has_frame:
                logger.info(f"no frame: {has_frame} {frame}")
                continue

            if frame is None:
                logger.info(f"frame empty: {has_frame} {frame}")
                continue

            logger.info(f"type(frame): {type(frame)}")
            frame_save_as_jpg(frame, fps)
            window_name = 'webcam'
            cv2.imshow(window_name, frame)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # FPS info
            logger.info("approx. FPS/elasped_time/#frames: {:.2f}/{:.2f}/{}".format(fps.fps(), fps.elapsed(), fps._numFrames))
            logger.info(f"cap: {cap is None}")
        else:
            cap.stream.retrieve()
        # update the FPS counter
        fps.update()

        # stop() method updates the _end attribute
        fps.stop()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("break")
            break


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