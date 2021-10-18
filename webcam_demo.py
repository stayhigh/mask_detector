#!/usr/bin/env python3
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from imutils.video import WebcamVideoStream  # threaded version

MASK_MODEL_ENABLE = True

SOURCE = 0  # or .mp4 file path
INPUT_SIZE = (112, 112) 
MODEL_PATH = 'exported/'



def crop_img(end_x, end_y, frame, start_x, start_y):
    face_img = frame[start_y:end_y, start_x:end_x, :]
    print ("INPUT_SIZE:", INPUT_SIZE)
    print ("face_img:", face_img)
    try:
        face_img = cv2.resize(face_img, INPUT_SIZE)
    except cv2.error as e:
        pass
    face_img = face_img - 127.5
    face_img = face_img * 0.0078125
    return face_img


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


def main():
    face_detector = MTCNN()
    if MASK_MODEL_ENABLE:
        mask_model = tf.keras.models.load_model(MODEL_PATH)

    # non-threaded source:    cap = cv2.VideoCapture(SOURCE)
    vs = WebcamVideoStream(src=SOURCE).start()
    cap = vs.stream

    have_mask = False # here
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # face detector is too slow
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

if __name__ == '__main__':
    main()
