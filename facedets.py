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


def main():
    # capture frames from a camera
    # cap = cv2.VideoCapture(args.src)
    cap = WebcamVideoStream(src=args.src).start()
    fps = FPS().start()
    fps.stop()

    while cv2.waitKey(1) < 0 and fps._numFrames < 10:
        # reads frames from a camera (cv2.VideoCapture)
        # ret, img = cap.read()
        # hasFrame, frame = cap.read()
        # if not hasFrame:
        #     cv2.waitKey()
        #     break
        frame = cap.read()

        # save frame for debugging
        # frame_save_as_jpg(frame, fps)

        # convert to gray scale of each frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detects faces of different sizes in the input image
        haarcascades_detector_faces = haarcascades_detector.detectMultiScale(gray, 1.3, 5)
        mtcnn_face_detector_faces = mtcnn_face_detector.detect_faces(frame) # slower than haar and YOLO_V5

        # haar_cascade model
        for haar in haarcascades_detector_faces:
            (x, y, w, h) = haar
            center = (int(x + w / 2), int(y + h / 2))
            radius = int(max(w, h) / 2)
            color = (255, 255, 0)
            thickness = 8
            cv2.circle(frame, center=center, radius=radius, color=color, thickness=thickness)
            cv2.putText(frame, 'HAAR_CASCADE ', org=(center[0] - radius, center[1] - radius), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

        # MTCNN model
        for face_loc in mtcnn_face_detector_faces:
            (x, y, w, h) = face_loc['box']
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 8)
            cv2.putText(frame, 'MTCNN', (x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color,
                        thickness=thickness)

        # YOLO v5 model
        frame, bboxes = getFaceBox(faceNet, frame)

        # Display an image in a window
        window_name = 'opencv face detection'
        cv2.imshow(window_name, frame)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # # Wait for Esc key to stop
        # if cv2.waitKey(10) & 0xFF == 27:
        #     break

        # update the FPS counter
        fps.update()

        # stop() method updates the _end attribute
        fps.stop()

        logger.info("approx. FPS/elasped_time/#frames: {:.2f}/{:.2f}/{}".format(fps.fps(), fps.elapsed(),
                                                                                           fps._numFrames))

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