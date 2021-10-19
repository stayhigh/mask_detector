# import the necessary packages
from __future__ import print_function


from imutils.video import WebcamVideoStream  # threaded version
from imutils.video import FPS
import argparse
import imutils
import cv2
import datetime


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

@profile
def set_camera_resolution_for_normal_fps(camera, width=args['width'], height=args['height']):
    print('setting video resolution:{:.0f}x{:.0f}'.format(width, height))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_video_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_video_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('actual video resolution:{:.0f}x{:.0f}'.format(actual_video_width, actual_video_height))
    return camera

@profile
def check_normal_fps():
    print("[INFO] sampling frames from webcam...")
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
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
        # update the FPS counter
        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    stream.release()
    cv2.destroyAllWindows()
    return fps.fps()

@profile
def check_threaded_fps():
    # created a *threaded* video stream, allow the camera sensor to warmup,
    # and start the FPS counter
    print("[INFO] sampling THREADED frames from webcam...")
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
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
        # update the FPS counter
        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    return fps.fps()


if __name__ == '__main__':
    print('--check_normal_fps--')
    normal_fps = check_normal_fps()

    print('--check_threaded_fps--')
    threaded_fps = check_threaded_fps()

    print("threaded fps is {:.3f} times faster than normal fps.".format(float(threaded_fps) / float(normal_fps)))
