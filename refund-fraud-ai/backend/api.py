from fastapi import FastAPI, UploadFile, File
import shutil
import tensorflow as tf
import cv2
import numpy as np
import os

app = FastAPI()

# ✅ Load trained model (correct path)
model = tf.keras.models.load_model("models/best_model.keras")

@app.post("/check-refund")
async def check_refund(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"

    # Save uploaded file
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read and preprocess image
    img = cv2.imread(temp_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prob = model.predict(img)[0][0]

    # Cleanup temp file
    os.remove(temp_path)

    result = "AI_GENERATED" if prob > 0.6 else "REAL_IMAGE"

    return {
        "result": result,
        "fraud_probability": round(float(prob), 3)
    }
