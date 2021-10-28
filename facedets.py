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

# set logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# parse arguments
parser = argparse.ArgumentParser(description='OpenCV Face Detection')
parser.add_argument('--src', action='store', default=0, nargs='?', help='Set video source; default is usb webcam')
parser.add_argument('--w', action='store', default=320, nargs='?', help='Set video width')
parser.add_argument('--h', action='store', default=240, nargs='?', help='Set video height')
parser.add_argument("--device", default="cpu", help="Device to inference on")
args = parser.parse_args()


# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.

# haarcascades, MTCNN, YOLOv5
haarcascades_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
haarcascades_eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') # https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
mtcnn_face_detector = MTCNN()

## YOLOv5
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

def  getFaceBox(net, frame, conf_threshold=0.7):
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
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 255), int(round(frameHeight/150)), 8)
            cv2.putText(frameOpencvDnn, 'YOLO_V5', (x1, y1 - padding), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
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
    mtcnn_face_detector_faces = mtcnn_face_detector.detect_faces(frame)  # slower than haar and YOLO_V5
    for face_loc in mtcnn_face_detector_faces:
        (x, y, w, h) = face_loc['box']
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 8)
        cv2.putText(frame, 'MTCNN', (x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color,
                    thickness=thickness)

def haarcascades_detect(haarcascades_detector, frame, haarcascades_eye_detector=None):
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
        thickness = 8
        cv2.circle(frame, center=center, radius=radius, color=color, thickness=thickness)
        cv2.putText(frame, 'HAAR_CASCADE ', org=(center[0] - radius, center[1] - radius),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # haar eyes
        if not haarcascades_eye_detector is None:
            eyes = haarcascades_eye_detector.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(roi_color, 'eyes ', org=(ex - 10, ey - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2)
def main():
    # capture frames from a camera
    # cap = cv2.VideoCapture(args.src)

    CAM_INDEX = args.src
    cap = WebcamVideoStream(src=CAM_INDEX + cv2.CAP_ANY).start()
    # cap = WebcamVideoStream(src=args.src).start()
    fps = FPS().start()
    fps.stop()

    # check camera is opened or not.
    if cap.stream == None or not cap.stream.isOpened():
        print('\n\n')
        print('Error - could not open video device.')
        print('\n\n')
        exit(0)

    logging.info(f"backend API: {cap.stream.getBackendName()}")

    while cv2.waitKey(1) < 0 and fps._numFrames < float("inf"):
        # reads frames from a camera (cv2.VideoCapture)
        # hasFrame, frame = cap.read()
        # if not hasFrame:
        #     cv2.waitKey()
        #     break

        frame = cap.read()

        # update the FPS counter
        fps.update()

        # check if the frame is empty
        if frame is None:
            logger.error(f"NO FRAME: frame type:{type(frame)}; frame repr: {repr(frame)}; _numFrames: {fps._numFrames}")
            continue

        # save frame for debugging
        # rotated_frame = rotate_picture(frame)
        # frame_save_as_jpg(rotated_frame, fps)

        # Haar Cascade
        haarcascades_detect(haarcascades_detector, frame, haarcascades_eye_detector=None)
        # MTCNN model
        mtcnn_detect(mtcnn_face_detector, frame)
        # YOLO v5 model
        frame, bboxes = getFaceBox(faceNet, frame)

        # FPS info
        logger.info("approx. FPS/elasped_time/#frames: {:.2f}/{:.2f}/{}".format(fps.fps(), fps.elapsed(), fps._numFrames))
        fps_info_xpadding = 20
        fps_info_ypadding = 20
        cv2.putText(frame,
                    f"FPS/elapsed time: {fps.fps():.4}/{fps.elapsed():.4}",
                    (0+fps_info_xpadding, 0+fps_info_ypadding),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(255, 0, 0),
                    thickness=2)

        # Display an image in a window
        window_name = 'opencv face detection'
        cv2.imshow(window_name, frame)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # # Wait for Esc key to stop
        # if cv2.waitKey(10) & 0xFF == 27:
        #     break

        # stop() method updates the _end attribute
        fps.stop()


    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # add function to be profiled
    lprofiler = LineProfiler()
    lprofiler.add_function(main)

    # set wrapper
    lp_wrapper = lprofiler(main)
    lp_wrapper()

    # save the output of profiling
    statfile = "{}.lprof".format(sys.argv[0])
    lprofiler.dump_stats(statfile)