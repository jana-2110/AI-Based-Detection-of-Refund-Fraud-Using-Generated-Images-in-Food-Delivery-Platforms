import cv2
import numpy as np
import tensorflow as tf
from backend.image_reuse import is_image_reused
from backend.fraud_score import calculate_fraud_score

MODEL_PATH = "model/refund_fraud_detector.keras"
IMAGE_PATH = "dataset/test/real/sample.jpg"  # change image here
IMG_SIZE = 224
THRESHOLD = 0.6

print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("❌ Image not found")

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)[0][0]
ai_prob = float(prediction)

print(f"\n🧠 AI Probability: {ai_prob:.2f}")

reused, diff = is_image_reused(IMAGE_PATH)
print("⚠️ Image reused" if reused else "✅ Image not reused")

fraud_score = calculate_fraud_score(
    ai_prob=ai_prob,
    reused=reused,
    exif_mismatch=False,
    user_history_score=0.3
)

print(f"\n🚨 Fraud Score: {fraud_score}/100")

if fraud_score >= 70:
    decision = "❌ REFUND DENIED"
elif fraud_score >= 40:
    decision = "⚠️ MANUAL REVIEW"
else:
    decision = "✅ REFUND APPROVED"

print(f"\n📌 FINAL DECISION: {decision}")
