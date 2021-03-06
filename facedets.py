# OpenCV program to detect face in real time
# import libraries of python OpenCV
# where its functionality resides
import sys
import cv2
import argparse
import os
import logging
from mtcnn_cv2 import MTCNN
from line_profiler import LineProfiler
from imutils.video import WebcamVideoStream  # threaded version
import imutils
from imutils.video import FPS
import logging
import time
import numpy as np
import math
import time

ENABLE_OUTPUT_VIDEO = True
ENABLE_DISPLAY = True
# set logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# parse arguments
parser = argparse.ArgumentParser(description='OpenCV Face Detection')
parser.add_argument('--src', action='store', default=0, nargs='?', help='Set video source; default is usb webcam')
parser.add_argument('--w', action='store', default=320, nargs='?', help='Set video width')
parser.add_argument('--h', action='store', default=240, nargs='?', help='Set video height')
parser.add_argument("--inputvideo", type=str, default="", help="set input video")
parser.add_argument("--nodisplay", action="store_true", default="", help="enable display")
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--model", required=True, default="dnn", help="enable all models: all/dnn")
args = parser.parse_args()

if args.nodisplay:
    ENABLE_DISPLAY = False

# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.


if args.model == "all":
    # haarcascades, MTCNN, DNN
    haarcascades_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    haarcascades_eye_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')  # https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
    mtcnn_face_detector = MTCNN()
    ## DNN
    faceProto = "/Users/johnwcwang/Desktop/codebase/learnopencv/AgeGender/opencv_face_detector.pbtxt"
    faceModel = "/Users/johnwcwang/Desktop/codebase/learnopencv/AgeGender/opencv_face_detector_uint8.pb"
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    if args.device == "cpu":
        faceNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif args.device == "gpu":
        faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")
elif args.model == "haar":
    haarcascades_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    haarcascades_eye_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')  # https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
elif args.model == 'mtcnn':
    mtcnn_face_detector = MTCNN()
elif args.model == "dnn":
    ## DNN
    faceProto = "/Users/johnwcwang/Desktop/codebase/learnopencv/AgeGender/opencv_face_detector.pbtxt"
    faceModel = "/Users/johnwcwang/Desktop/codebase/learnopencv/AgeGender/opencv_face_detector_uint8.pb"
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    if args.device == "cpu":
        faceNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif args.device == "gpu":
        faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")
elif args.model == "yolo":
    yolo_net = cv2.dnn.readNetFromDarknet('./yolov3.cfg', './yolov3.weights')
    # yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def dnn_getFaceBox(net, frame, conf_threshold=0.7):
    thickness = 2
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    padding = 10
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            # cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 255), int(round(frameHeight / 150)), 2)
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 255), thickness=thickness, lineType=2)
            cv2.putText(frameOpencvDnn, 'DNN', (x1, y1 - padding), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
                        thickness=thickness, lineType=cv2.LINE_AA)
    return frameOpencvDnn, bboxes


def yolo_getBox(net, frame, conf_threshold=0.5):
    """
    https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
    :param net:
    :param frame:
    :param conf_threshold:
    :return:
    """
    classes = open('coco.names').read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    ln = net.getLayerNames()
    # print(len(ln), ln)
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    padding = 10
    r = blob[0, 0, :, :]
    # text = f'Blob shape={blob.shape}'
    # cv2.displayOverlay('blob', text)
    if ENABLE_DISPLAY:
        cv2.imshow('blob', r)

    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time()

    # The forward propagation takes about 2 seconds on an MacAir 2012 (1,7 GHz Intel Core i5).
    # https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
    print(f'forward propagation time={t - t0}')

    bboxes = []
    confidences = []
    classIDs = []
    h, w = frameOpencvDnn.shape[:2]
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf_threshold:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                bboxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(bboxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (bboxes[i][0], bboxes[i][1])
            (w, h) = (bboxes[i][2], bboxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            # thickness = math.ceil(10 * (confidences[i] - conf_threshold) / conf_threshold)
            cv2.rectangle(frameOpencvDnn, (x, y), (x + w, y + h), color, thickness=1)
            # cv2.rectangle(frameOpencvDnn, (x, y), (x + w, y + h), color, int(round(frameHeight / 150)), 8)
            # cv2.rectangle(frameOpencvDnn, (x, y), (x + w, y + h), color, int(round(frameHeight / 150)),
            #               thickness=8 * (1 + int(confidences[i])))
            text = "{}: {:.3f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(frameOpencvDnn, f'YOLO-{text}', (x, y - padding), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                        cv2.LINE_AA)

    return frameOpencvDnn, bboxes


def frame_save_as_jpg(frame, fps):
    resized_frame = imutils.resize(frame, width=400)
    directory = os.path.join(os.path.dirname(__file__), './jpgs')
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(f"jpgs/{fps._numFrames}.jpg", resized_frame)


def rotate_picture(image, width=400, height=400, angle=45, scale=1):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    # using cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
    return rotated_image


def mtcnn_detect(mtcnn_face_detector, frame):
    # MTCNN model
    thickness = 2
    mtcnn_face_detector_faces = mtcnn_face_detector.detect_faces(frame)  # slower than haar and DNN
    for face_loc in mtcnn_face_detector_faces:
        (x, y, w, h) = face_loc['box']
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=thickness)
        cv2.putText(frame, 'MTCNN', (x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color,
                    thickness=thickness)


def haarcascades_detect(haarcascades_detector, frame, haarcascades_eye_detector=None):
    thickness = 2
    # convert to gray scale of each frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # Detects faces of different sizes in the input image
    haarcascades_detector_faces = haarcascades_detector.detectMultiScale(gray, 1.3, 5)

    # # haar_cascade model
    for haar in haarcascades_detector_faces:
        (x, y, w, h) = haar
        center = (int(x + w / 2), int(y + h / 2))
        radius = int(max(w, h) / 2)
        color = (255, 255, 0)

        cv2.circle(frame, center=center, radius=radius, color=color, thickness=thickness)
        cv2.putText(frame, 'HAAR_CASCADE ', org=(center[0] - radius, center[1] - radius),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=thickness)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # haar eyes
        if not haarcascades_eye_detector is None:
            eyes = haarcascades_eye_detector.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(roi_color, 'eyes ', org=(ex - 10, ey - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=1)


def main():
    # capture frames from a camera
    logger.info(f"inputvideo:{args.inputvideo}")
    if not args.inputvideo:
        CAM_INDEX = args.src
        logging.info(f"{CAM_INDEX} {cv2.CAP_ANY}")
        #cap = WebcamVideoStream(src=CAM_INDEX + cv2.CAP_ANY).start()
        cap = cv2.VideoCapture(CAM_INDEX + cv2.CAP_ANY)
        # check camera is opened or not.
        if cap == None or not cap.isOpened():
            logger.error('\n\nError - could not open video device.\n\n')
            exit(0)
        logging.info(f"backend API: {cap.getBackendName()}")
    else:
        # cap = WebcamVideoStream(src=args.inputvideo).start()
        cap = cv2.VideoCapture(args.inputvideo)
        # check camera is opened or not.
        if cap == None or not cap.isOpened():
            logger.error('\n\nError - could not open video device.\n\n')
            exit(0)
        logging.info(f"backend API: {cap.getBackendName()}")

    fps = FPS().start()
    fps.stop()

    # enable video writer
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    if ENABLE_OUTPUT_VIDEO:
        if not args.inputvideo:
            outputfilename = "webcam"
        else:
            outputfilename = args.inputvideo
        ts = int(time.time())
        output_video = cv2.VideoWriter(f'output_{outputfilename}_{ts}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
        if not output_video.isOpened():
            logger.warning(f"VideoWriter is not opened.")
            sys.exit(-1)

    while cv2.waitKey(1) < 0 and fps._numFrames < float("inf"):
        # reads frames from a camera (cv2.VideoCapture)
        # hasFrame, frame = cap.read()
        # if not hasFrame:
        #     cv2.waitKey()
        #     break
        ret, frame = cap.read()

        if not ret:
            logger.warning(f"ret error on #frame {fps._numFrames}")
            continue
        # update the FPS counter
        fps.update()

        # check if the frame is empty
        if frame is None:
            logger.error(f"NO FRAME: frame type:{type(frame)}; frame repr: {repr(frame)}; _numFrames: {fps._numFrames}")
            break

        # save frame for debugging
        # rotated_frame = rotate_picture(frame)
        # frame_save_as_jpg(rotated_frame, fps)

        if args.model == "all":
            # Haar Cascade
            haarcascades_detect(haarcascades_detector, frame, haarcascades_eye_detector=None)
            # MTCNN model
            mtcnn_detect(mtcnn_face_detector, frame)
            # dnn model
            frame, bboxes = dnn_getFaceBox(faceNet, frame)
        elif args.model == "haar":
            haarcascades_detect(haarcascades_detector, frame, haarcascades_eye_detector=None)
        elif args.model == "mtcnn":
            mtcnn_detect(mtcnn_face_detector, frame)
        elif args.model == "dnn":
            # dnn model
            frame, bboxes = dnn_getFaceBox(faceNet, frame)
        elif args.model == 'yolo':
            """yolo for coco.name object detection"""
            frame, bboxes = yolo_getBox(yolo_net, frame)

        # FPS info
        logger.info(
            "approx. FPS/elasped_time/#frames: {:.2f}/{:.2f}/{}".format(fps.fps(), fps.elapsed(), fps._numFrames))
        fps_info_xpadding = 20
        fps_info_ypadding = 20
        cv2.putText(frame,
                    f"FPS/elapsed time: {fps.fps():.4}/{fps.elapsed():.4}",
                    (0 + fps_info_xpadding, 0 + fps_info_ypadding),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(255, 0, 0),
                    thickness=2)

        # Display an image in a window
        window_name = 'opencv face detection'
        if ENABLE_DISPLAY:
            cv2.imshow(window_name, frame)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # output video
        if ENABLE_OUTPUT_VIDEO:
            output_video.write(frame)

        # # Wait for Esc key to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # stop() method updates the _end attribute
        fps.stop()

    # release the video
    output_video.release()

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


def test1():
    import cv2
    haarcascades_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    haarcascades_eye_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')  # https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
    mtcnn_face_detector = MTCNN()
    ## DNN
    faceProto = "/Users/johnwcwang/Desktop/codebase/learnopencv/AgeGender/opencv_face_detector.pbtxt"
    faceModel = "/Users/johnwcwang/Desktop/codebase/learnopencv/AgeGender/opencv_face_detector_uint8.pb"

    # importing libraries
    import cv2
    import numpy as np

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture('ipman.mp4')

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Haar Cascade
            # haarcascades_detect(haarcascades_detector, frame, haarcascades_eye_detector=None)
            # MTCNN model
            # mtcnn_detect(mtcnn_face_detector, frame)
            # dnn model
            # frame, bboxes = dnn_getFaceBox(faceNet, frame)
            frame, bboxes = yolo_getBox(yolo_net, frame)

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # add function to be profiled
    lprofiler = LineProfiler()
    lprofiler.add_function(main)
    lprofiler.add_function(dnn_getFaceBox)
    lprofiler.add_function(yolo_getBox)

    # set wrapper
    lp_wrapper = lprofiler(main)
    lp_wrapper()

    # save the output of profiling
    statfile = "{}.lprof".format(sys.argv[0])
    lprofiler.dump_stats(statfile)
