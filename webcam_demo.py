#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from imutils.video import WebcamVideoStream  # threaded version
from imutils.video import FPS
from line_profiler import LineProfiler
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--timeout", type=int, default=30,
                help="timeout (seconds)")
ap.add_argument("-n", "--num-frames", type=int, default=float("inf"),
                help="# of frames to loop over for FPS test")
args = vars(ap.parse_args())

MASK_MODEL_ENABLE = False

SOURCE = 0  # or .mp4 file path
INPUT_SIZE = (112, 112) 
MODEL_PATH = 'exported/'


# @profile
def crop_img(end_x, end_y, frame, start_x, start_y):
    face_img = frame[start_y:end_y, start_x:end_x, :]
    # print ("INPUT_SIZE:", INPUT_SIZE)
    # print ("face_img:", face_img)
    try:
        face_img = cv2.resize(face_img, INPUT_SIZE)
    except cv2.error as e:
        pass
    face_img = face_img - 127.5
    face_img = face_img * 0.0078125
    return face_img

# @profile
def draw_bbox(frame, start_x, start_y, end_x, end_y, have_mask):
    # frame now is RGB
    #color = (0, 255, 0) if have_mask else (255, 0, 0)
    if have_mask:
        color = (0, 255, 0)
        cv2.rectangle(frame,
                  (start_x, start_y),
                  (end_x, end_y),
                  color,
                  2)
        cv2.putText(frame, 'mask_protected', (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    else:
        color = (255, 0, 0)
        cv2.rectangle(frame,
                  (start_x, start_y),
                  (end_x, end_y),
                  color,
                  2)
        cv2.putText(frame, 'no_mask', (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame

# @profile
def main():
    face_detector = MTCNN()
    if MASK_MODEL_ENABLE:
        mask_model = tf.keras.models.load_model(MODEL_PATH)

    # non-threaded source:    cap = cv2.VideoCapture(SOURCE)
    vs = WebcamVideoStream(src=SOURCE).start()
    fps = FPS().start()
    cap = vs.stream

    have_mask = False # here

    while fps._numFrames < args["num_frames"]:
        print('num of frames: {}'.format(fps._numFrames))
        if getattr(fps, '_end') and fps.elapsed() > args["timeout"]:
            print("timeout {} seconds".format(args["timeout"]))
            break
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # face detector is too slow
        lprofiler.add_function(face_detector.detect_faces)
        face_locs = face_detector.detect_faces(frame)

        for face_loc in face_locs:
            bbox = face_loc['box']
            start_x = bbox[0]
            start_y = bbox[1]
            end_x = bbox[0] + bbox[2]
            end_y = bbox[1] + bbox[3]

            face_img = crop_img(end_x, end_y, frame, start_x, start_y)

            if MASK_MODEL_ENABLE:
                try:
                    mask_result = mask_model.predict(np.expand_dims(face_img, axis=0))[0]
                    have_mask = np.argmax(mask_result)
                except:
                    print ('err_on_predict')

            frame = draw_bbox(frame, start_x, start_y, end_x, end_y, have_mask)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        window_name = 'mask detector'
        cv2.imshow(window_name, frame)

        # set the window on topmost
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fps.update()
        fps.stop()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    lprofiler = LineProfiler()
    lprofiler.add_function(main)
    lprofiler.add_function(crop_img)
    lprofiler.add_function(draw_bbox)

    lp_wrapper = lprofiler(main)
    lp_wrapper()
    statfile = "{}.lprof".format(sys.argv[0])
    lprofiler.dump_stats(statfile)
    # main()
