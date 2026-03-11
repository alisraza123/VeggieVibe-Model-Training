import tensorflow as tf
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

train_dir = "train"
val_dir = "val"
test_dir = "test"

print("🔹 Stage 1: Preparing datasets...")

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = val_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

NUM_CLASSES = train_data.num_classes

print(f"✅ Stage 1 Done: Found {NUM_CLASSES} classes")

# MobileNetV2 model
print("🔹 Stage 2: Building model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Stage 2 Done: Model built successfully")

# Callback for time tracking
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.train_start_time = time.time()
        print(f"🔹 Stage 3: Training started at {time.ctime(self.train_start_time)}")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\n➡️ Starting Epoch {epoch+1}/{EPOCHS}...")

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        avg_time = np.mean(self.epoch_times)
        remaining_epochs = EPOCHS - (epoch + 1)
        est_remaining = remaining_epochs * avg_time
        print(f"✅ Epoch {epoch+1} finished in {epoch_time:.2f}s")
        print(f"Estimated remaining time: {est_remaining/60:.2f} minutes")

time_callback = TimeHistory()

# Training
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[time_callback]
)

total_time = time.time() - time_callback.train_start_time
print(f"\n✅ Training finished in {total_time/60:.2f} minutes")

# Save model
print("🔹 Stage 4: Saving model...")
model.save("model.h5")
print("✅ Model saved as model.h5")

# Accuracy Graph
print("🔹 Stage 5: Plotting accuracy and loss graphs...")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"])
plt.savefig("accuracy_graph.png")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"])
plt.savefig("loss_graph.png")
plt.show()
print("✅ Accuracy & Loss graphs saved")

# Predictions & Confusion Matrix
print("🔹 Stage 6: Evaluating on test set...")
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()
print("✅ Confusion matrix saved")

# Classification Report
class_names = list(test_data.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:\n")
print(report)
with open("classification_report.txt","w") as f:
    f.write(report)
print("✅ Classification report saved as classification_report.txt")