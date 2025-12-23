import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model("../models/best_model.keras")

test_dir = "../dataset/test"

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=1,
    class_mode="binary",
    shuffle=False
)

pred_probs = model.predict(test_data)
pred_labels = (pred_probs > 0.5).astype(int)

true_labels = test_data.classes

cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:")
print(classification_report(
    true_labels,
    pred_labels,
    target_names=["fake", "real"]
))
