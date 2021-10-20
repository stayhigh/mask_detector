#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from imutils.video import WebcamVideoStream  # threaded version
from imutils.video import FPS
import imutils
from line_profiler import LineProfiler
import argparse
import traceback
import logging
from threading import Thread
from queue import Queue
import multiprocessing
import multiprocessing.queues
import time
import concurrent.futures


# set logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", type=int, default=1,
                help="Whether or not frames should be displayed")
ap.add_argument("-t", "--timeout", type=int, default=90,
                help="timeout (seconds)")
ap.add_argument("-n", "--num-frames", type=int, default=float("inf"),
                help="# of frames to loop over for FPS test")
ap.add_argument("-p", "--procs", type=int, default=-1,
		help="# of processes to spin up")
args = vars(ap.parse_args())

# check cpu counts
procs = args["procs"] if args["procs"] > 0 else multiprocessing.cpu_count()
procIDs = list(range(0, procs))
logging.info(f"procIDs:{str(procIDs)}")

# MODEL
MASK_MODEL_ENABLE = False
FACE_MODEL_ENABLE = False

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


def push_frame_queue(q):
    logger.info(f"[push_frame_queue] running")
    vs = WebcamVideoStream(src=SOURCE).start()
    fps = FPS().start()
    # cap = vs.stream # too slow, but it support different frame objects
    fps.stop() # enable the _end attribute

    while fps._numFrames < args["num_frames"]:
        if fps.elapsed() > args["timeout"]:
            logger.info("timeout {} seconds".format(args["timeout"]))
            break
        # ret, frame = cap.read() # too slow, but it support different frame objects
        frame = vs.read()
        # frame = imutils.resize(frame, width=400)
        q.put(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # update the FPS counter
        fps.update()
        fps.stop()
        logger.info("[push_frame_queue] approx. elasped_time/FPS: {:.2f}/{:.2f}".format(fps.elapsed(), fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


def get_frame_queue(input_queue):
    logger.info(f"[get_frame_queue] running")
    while True:
        frame = input_queue.get()
        if FACE_MODEL_ENABLE:
            global face_detector
            face_locs = face_detector.detect_faces(frame)
            have_mask = False
            for face_loc in face_locs:
                bbox = face_loc['box']
                start_x = bbox[0]
                start_y = bbox[1]
                end_x = bbox[0] + bbox[2]
                end_y = bbox[1] + bbox[3]

                face_img = crop_img(end_x, end_y, frame, start_x, start_y)

                if MASK_MODEL_ENABLE:
                    global mask_model
                    try:
                        mask_result = mask_model.predict(np.expand_dims(face_img, axis=0))[0]
                        have_mask = np.argmax(mask_result)
                    except Exception as e:
                        logger.error('err_on_mask_model_predict')
                        logger.exception(e)
                        logger.debug(traceback.format_exc())
                        traceback.print_exc()

                frame = draw_bbox(frame, start_x, start_y, end_x, end_y, have_mask)
        # set the window on topmost
        window_name = 'mask detector (Consumer)'
        cv2.imshow(window_name, frame)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()


# global
logger.info("load model (global)")
face_detector = MTCNN()
logger.info("loaded face_detector: MTCNN model")
mask_model = tf.keras.models.load_model(MODEL_PATH)
logger.info("loaded mask_model: keras model")

def main():
    logger.info("Producer (Daemon): push frame into multiprocessing.Queue")
    pure_queue = multiprocessing.Queue(1) # large queue size cause long display latency
    push_p = multiprocessing.Process(name='ProducerProcess', target=push_frame_queue, args=(pure_queue,), daemon=True)
    push_p.start()

    logger.info("Consumer (Daemon): get frame into multiprocessing.Queue")
    get_p = multiprocessing.Process(name='ConsumerProcess', target=get_frame_queue, args=(pure_queue, ), daemon=True)
    get_p.start()

    logger.info("join processes: Producer, Consumer, and Display")
    push_p.join()
    get_p.join()

if __name__ == '__main__':
    # add function to be profiled
    lprofiler = LineProfiler()
    lprofiler.add_function(main)
    lprofiler.add_function(crop_img)
    lprofiler.add_function(draw_bbox)

    # set wrapper
    lp_wrapper = lprofiler(main)
    lp_wrapper()

    # output profile file
    statfile = "{}.lprof".format(sys.argv[0])
    lprofiler.dump_stats(statfile)
