# OpenCV program to detect face in real time
# import libraries of python OpenCV
# where its functionality resides
import cv2
import argparse
import os
import logging
from mtcnn_cv2 import MTCNN
from line_profiler import LineProfiler
from imutils.video import WebcamVideoStream  # threaded version

# set logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# parse arguments
parser = argparse.ArgumentParser(description='OpenCV Face Detection')
parser.add_argument('--src', action='store', default=0, nargs='?', help='Set video source; default is usb webcam')
parser.add_argument('--w', action='store', default=320, nargs='?', help='Set video width')
parser.add_argument('--h', action='store', default=240, nargs='?', help='Set video height')
args = parser.parse_args()

# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.
haarcascades_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mtcnn_face_detector = MTCNN()

def main():
    # capture frames from a camera
    # cap = cv2.VideoCapture(args.src)
    cap = WebcamVideoStream(src=args.src).start()

    while True:
        # reads frames from a camera
        # ret, img = cap.read()
        img = cap.read()

        # convert to gray scale of each frames
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detects faces of different sizes in the input image
        haarcascades_detector_faces = haarcascades_detector.detectMultiScale(gray, 1.3, 5)
        mtcnn_face_detector_faces = mtcnn_face_detector.detect_faces(img)

        # haar_cascade
        for (x, y, w, h) in haarcascades_detector_faces:
            center = (int(x + w / 2), int( y + h / 2))
            radius = int(max(w, h) / 2)
            color = (255, 255, 0)
            thickness = 2
            cv2.circle(img, center=center, radius=radius, color=color, thickness=thickness)
            cv2.putText(img, 'haar_cascade', org=(center[0] - radius, center[1] - radius), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=color, thickness=2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

        # mtcnn
        for face_loc in mtcnn_face_detector_faces:
            (x, y, w, h) = face_loc['box']
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, 'mtcnn', (x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=color, thickness=thickness)

        # Display an image in a window
        window_name = 'opencv face detection'
        cv2.imshow(window_name, img)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Wait for Esc key to stop
        if cv2.waitKey(10) & 0xFF == 27:
            break

    # Close the window
    cap.release()

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # add function to be profiled
    lprofiler = LineProfiler()
    lprofiler.add_function(main)

    # set wrapper
    lp_wrapper = lprofiler(main)
    lp_wrapper()
