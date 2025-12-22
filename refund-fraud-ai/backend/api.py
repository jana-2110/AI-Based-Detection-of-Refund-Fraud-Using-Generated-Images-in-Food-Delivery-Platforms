from fastapi import FastAPI, UploadFile, File
import shutil
import tensorflow as tf
import cv2
import numpy as np

app = FastAPI()
model = tf.keras.models.load_model("model/detector.h5")

@app.post("/check-refund")
async def check_refund(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    result = "AI_GENERATED" if pred > 0.6 else "REAL_IMAGE"

    return {
        "result": result,
        "confidence": float(pred)
    }
