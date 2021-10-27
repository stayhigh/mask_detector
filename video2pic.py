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

# set logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


cap = WebcamVideoStream(src=0).start()
fps = FPS().start()
fps.stop()

def frame_save_as_jpg(frame, fps):
    # frame = imutils.resize(frame, width=400)
    directory = os.path.join(os.path.dirname(__file__), './jpgs')
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(f"jpgs/{fps._numFrames}.jpg", frame)

MAX_FRAME_NUM = 10

while cv2.waitKey(1) < 0 and fps._numFrames < MAX_FRAME_NUM:
    frame = cap.read()
    frame_save_as_jpg(frame,fps)

    # FPS info
    logger.info("approx. FPS/elasped_time/#frames: {:.2f}/{:.2f}/{}".format(fps.fps(), fps.elapsed(), fps._numFrames))

    # update the FPS counter
    fps.update()

    # stop() method updates the _end attribute
    fps.stop()

# De-allocate any associated memory usage
cv2.destroyAllWindows()

