import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("model/detector.h5")

def is_ai_generated(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]
    return prob, prob > 0.8
