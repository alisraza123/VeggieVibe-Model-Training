import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Config
MODEL_PATH = "./Models/model.h5"       # trained model path
IMAGE_PATH = "image.jpg"      # new image for prediction
IMG_SIZE = 224                # same size as used in training

# Load model
print("🔹 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Load labels from labels.txt
with open("labels.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f]

print(f"🔹 Loaded class labels: {CLASS_NAMES}")


# Load and preprocess image
print(f"🔹 Loading image: {IMAGE_PATH}")
img = image.load_img(IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # batch dimension
img_array = img_array / 255.0  # rescale as in training


# Predict
print("🔹 Predicting...")
preds = model.predict(img_array)[0]

# Get top 3 predictions
top3_idx = preds.argsort()[-3:][::-1]
print("\n🔹 Top 3 Predictions:")
for i in top3_idx:
    print(f"{CLASS_NAMES[i]}: {preds[i]:.4f}")

# Final predicted class
predicted_class = CLASS_NAMES[np.argmax(preds)]
print(f"\n✅ Final Predicted class: {predicted_class}")

# Show image
plt.imshow(img)
plt.title(f"Prediction: {predicted_class}")
plt.axis("off")
plt.show()