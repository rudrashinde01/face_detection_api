from fastapi import FastAPI, UploadFile
import numpy as np
import cv2
from face_detect import detect_face

app = FastAPI()

@app.post("/detect-face")
async def detect(file: UploadFile):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    face_detected, confidence = detect_face(frame)

    if face_detected:
        return {
            "status": "approved",
            "confidence": confidence
        }

    return {
        "status": "denied",
        "confidence": 0
    }