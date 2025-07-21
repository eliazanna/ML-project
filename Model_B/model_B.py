import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import csv
import shutil
from PIL import Image

img_height, img_width = 200, 300
batch_size = 32 
epochs = 20  #con early stopping lo fermerà prima
base_dir = "D:/DESKTOP/Desktop/ML-project/Model_B/rps-clean"

#dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    validation_split=0.2 ,
    subset='training',
    seed=42 ,
    shuffle=True
)

val_data = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    validation_split=0.2 ,
    subset='validation',
    seed=42
)

#data augmentation su training
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomBrightness(0.2),
])
#rescaling
train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True) / 255.0, y))
val_data = val_data.map(lambda x, y: (x / 255.0, y))

#Modello CNN, tre layers
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),  # layer extra
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),  # per ridurre overfitting
    layers.Dense(3, activation='softmax')
])

#compilation e early stopping
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# === Training ===
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stop]
)

#plotto rapporto tra accuracy test e train
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model B - Accuracy migliorata")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#salvo il predittore
model.save("D:/DESKTOP/Desktop/ML-project/Model_B/predictor_B.keras")

#analisi errori sul validation set
WRONG_DIR = "D:/DESKTOP/Desktop/ML-project/Model_B/wrong_predictions"
WRONG_IMG_DIR = os.path.join(WRONG_DIR, "images")
CLASS_NAMES = ['paper', 'rock', 'scissors']

# reset cartelle
if os.path.exists(WRONG_DIR):
    shutil.rmtree(WRONG_DIR)
os.makedirs(WRONG_IMG_DIR, exist_ok=True)

rows = []
counter = 1

# scorri immagini del validation set
for image, label in val_data.unbatch():
    img_array = image.numpy()
    true_idx = int(np.argmax(label.numpy()))

    pred = model.predict(img_array[np.newaxis, ...], verbose=0)[0]
    pred_idx = int(np.argmax(pred))

    if pred_idx != true_idx:
        # salva immagine
        filename = f"wrong_{counter:03d}.jpg"
        path = os.path.join(WRONG_IMG_DIR, filename)
        img_uint8 = (img_array * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(path)

        rows.append([filename, CLASS_NAMES[true_idx], CLASS_NAMES[pred_idx]])
        counter += 1

# salva CSV
csv_path = os.path.join(WRONG_DIR, "wrong_preds.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "true_label", "predicted_label"])
    writer.writerows(rows)

print(f"Salvati {len(rows)} errori del validation set in:\n- {WRONG_IMG_DIR}\n- CSV: {csv_path}")
