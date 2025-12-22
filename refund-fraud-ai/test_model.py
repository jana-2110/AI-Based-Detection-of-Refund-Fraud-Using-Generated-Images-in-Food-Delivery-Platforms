import cv2
import numpy as np
import tensorflow as tf
from backend.image_reuse import is_image_reused
from backend.fraud_score import calculate_fraud_score

# =========================
# CONFIG
# =========================
MODEL_PATH = "model/refund_fraud_detector.keras"
IMAGE_PATH = "dataset/test/real/1000138967.jpg"  # change image here
IMG_SIZE = 224
THRESHOLD = 0.6   # lowered threshold

# =========================
# LOAD MODEL
# =========================
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# =========================
# LOAD IMAGE
# =========================
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError("❌ Image not found or path incorrect")

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# =========================
# MODEL PREDICTION
# =========================
prediction = model.predict(img)[0][0]
ai_probability = float(prediction)

print(f"\n🧠 AI Probability: {ai_probability:.2f}")

# =========================
# IMAGE REUSE CHECK
# =========================
reused, diff = is_image_reused(IMAGE_PATH)

if reused:
    print(f"⚠️ Image Reuse Detected (hash diff = {diff})")
else:
    print("✅ Image is not reused")

# =========================
# FRAUD SCORE
# =========================
fraud_score = calculate_fraud_score(
    ai_prob=ai_probability,
    reused=reused,
    exif_mismatch=False,      # future extension
    user_history_score=0.3    # mock value
)

print(f"\n🚨 Fraud Score: {fraud_score}/100")

# =========================
# FINAL DECISION
# =========================
if fraud_score >= 70:
    decision = "❌ REFUND DENIED (HIGH FRAUD RISK)"
elif fraud_score >= 40:
    decision = "⚠️ MANUAL REVIEW REQUIRED"
else:
    decision = "✅ REFUND APPROVED"

print(f"\n📌 FINAL DECISION: {decision}")
