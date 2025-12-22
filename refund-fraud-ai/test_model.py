import tensorflow as tf
import cv2
import numpy as np
import os

print("Loading model...")
model = tf.keras.models.load_model("model/detector.h5")
print("✅ Model loaded successfully")

folder = "dataset/train/real"
files = os.listdir(folder)

if len(files) == 0:
    print("❌ No images found in", folder)
    exit()

image_path = os.path.join(folder, files[0])
print("Using image:", image_path)

img = cv2.imread(image_path)

if img is None:
    print("❌ OpenCV could not read the image")
    exit()

img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]

THRESHOLD = 0.6  # 🔥 lowered from 0.8

if pred > THRESHOLD:
    print("❌ AI GENERATED IMAGE")
else:
    print("✅ REAL IMAGE")

print("Confidence:", float(pred))
