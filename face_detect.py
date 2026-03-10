import cv2
import numpy as np

model = "res10_300x300_ssd_iter_140000.caffemodel"
config = "deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(config, model)

def detect_face(frame):

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame,(300,300)),
        1.0,
        (300,300),
        (104,177,123)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0,0,i,2]

        if confidence > 0.7:
            return True, float(confidence)

    return False, 0.0