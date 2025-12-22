import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -----------------------
# CONFIGURATION
# -----------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"

# -----------------------
# DATA GENERATORS
# -----------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# -----------------------
# LOAD BASE MODEL
# -----------------------
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# -----------------------
# FINE-TUNING STRATEGY
# -----------------------
base_model.trainable = True

# Freeze early layers, train deeper layers
for layer in base_model.layers[:200]:
    layer.trainable = False

# -----------------------
# BUILD FINAL MODEL
# -----------------------
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# -----------------------
# COMPILE MODEL
# -----------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------
# CALLBACKS
# -----------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6
)

# -----------------------
# TRAIN MODEL
# -----------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# -----------------------
# SAVE MODEL
# -----------------------
model.save("model/detector.h5")

print("✅ Training complete. Model saved as model/detector.h5")
