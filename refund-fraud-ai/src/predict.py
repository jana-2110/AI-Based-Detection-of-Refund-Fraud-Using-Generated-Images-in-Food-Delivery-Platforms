import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("../models/best_model.keras")

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]

    if prob > 0.7:
        decision = "⚠️ HIGH FRAUD RISK (Manual Review)"
    elif prob > 0.4:
        decision = "⚠️ MEDIUM RISK"
    else:
        decision = "✅ LOW RISK (Genuine)"

    print(f"Fraud Probability: {prob:.2f}")
    print(f"Decision: {decision}")

# Example
predict_image("sample_test.jpg")
