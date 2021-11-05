# import the necessary packages
from __future__ import print_function

import sys
from imutils.video import WebcamVideoStream  # threaded version
from imutils.video import FPS
import argparse
import imutils
import cv2
from line_profiler import LineProfiler
import logging

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
                help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
                help="Whether or not frames should be displayed")
ap.add_argument("--width", type=int, default=1280,
                help="the width of frames")
ap.add_argument("--height", type=int, default=720,
                help="the height of frames")
args = vars(ap.parse_args())

# set logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_normal_fps():
    logger.info("[INFO] sampling frames from webcam...")
    stream = cv2.VideoCapture(0)
    stream = set_camera_resolution_for_normal_fps(stream)
    fps = FPS().start()
    # loop over some frames
    while fps._numFrames < args["num_frames"]:
        # grab the frame from the stream and resize it to have a maximum
        # width of 400 pixels
        (grabbed, frame) = stream.read()
        frame = imutils.resize(frame, width=400)
        # check to see if the frame should be displayed to our screen
        if args["display"] > 0:
            window_name = "Normal Frame"
            cv2.imshow(window_name, frame)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            key = cv2.waitKey(1) & 0xFF
        # update the FPS counter
        fps.update()
        # stop the timer and display FPS information
        fps.stop()
        logger.info("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        logger.info("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    stream.release()
    cv2.destroyAllWindows()
    return fps.fps()

def set_camera_resolution_for_normal_fps(camera, width=args['width'], height=args['height']):
    logger.info('setting video resolution:{:.0f}x{:.0f}'.format(width, height))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_video_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_video_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info('actual video resolution:{:.0f}x{:.0f}'.format(actual_video_width, actual_video_height))
    return camera

def check_threaded_fps_withqueue(myqueue):
    # created a *threaded* video stream, allow the camera sensor to warmup,
    # and start the FPS counter
    logger.info("[INFO] sampling THREADED frames from webcam...")
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    # loop over some frames...this time using the threaded stream
    while fps._numFrames < args["num_frames"]:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        # check to see if the frame should be displayed to our screen
        if args["display"] > 0:
            window_name = "THREADED Frame"
            cv2.imshow(window_name, frame)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            key = cv2.waitKey(1) & 0xFF
        # update the FPS counter
        fps.update()
        myqueue.put(frame)
        # stop the timer and display FPfS information
        fps.stop()
        logger.info("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        logger.info("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    return fps.fps()

def check_threaded_fps():
    # created a *threaded* video stream, allow the camera sensor to warmup,
    # and start the FPS counter
    logger.info("[INFO] sampling THREADED frames from webcam...")
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    # loop over some frames...this time using the threaded stream
    while fps._numFrames < args["num_frames"]:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        # check to see if the frame should be displayed to our screen
        if args["display"] > 0:
            window_name = "THREADED Frame"
            cv2.imshow(window_name, frame)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            key = cv2.waitKey(1) & 0xFF
        # update the FPS counter
        fps.update()
        # stop the timer and display FPS information
        fps.stop()
        logger.info("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        logger.info("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    return fps.fps()

def main():
    logger.info('--check_normal_fps--')
    normal_fps = check_normal_fps()

    logger.info('--check_threaded_fps--')
    threaded_fps = check_threaded_fps()

    logger.info(f'check_threaded_fps/check_threaded_fps: {threaded_fps}/{normal_fps}')
    logger.info("threaded fps is {:.3f} times faster than normal fps.".format(float(threaded_fps) / float(normal_fps)))

if __name__ == '__main__':
    lprofiler = LineProfiler()
    lprofiler.add_function(check_normal_fps)
    lprofiler.add_function(check_threaded_fps)
    lprofiler.add_function(main)
    lp_wrapper = lprofiler(main)
    lp_wrapper()
    statfile = "{}.lprof".format(sys.argv[0])
    lprofiler.dump_stats(statfile)
    # lprofiler.print_stats()
