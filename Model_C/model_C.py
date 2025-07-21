import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import csv
import shutil
from PIL import Image

IMG_HEIGHT, IMG_WIDTH = 200, 300
EPOCHS = 20
BASE_DIR = "D:/DESKTOP/Desktop/ML-project/Model_C/rps-processed"
MODEL_SAVE_PATH = "D:/DESKTOP/Desktop/ML-project/Model_C/predictor_C.keras"
WRONG_DIR = "D:/DESKTOP/Desktop/ML-project/Model_C/wrong_predictions"
CLASS_NAMES = ['paper', 'rock', 'scissors']
BATCH_OPTIONS = [16, 32, 64]

#data aug
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomBrightness(0.4),
    layers.RandomContrast(0.4)
])

#FAST HYPERPARAMETER TUNING (BATCH SIZE)
#solo su 3 epoche: valore indicativo: evitiamo richieda molto tempo
TUNING_EPOCHS = 5
best_batch = None
best_val_acc = 0

for batch_size in BATCH_OPTIONS:
    print(f"\n🔍 Testing batch size: {batch_size}")

    train_data = tf.keras.utils.image_dataset_from_directory(
        BASE_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        validation_split=0.2,
        subset='training',
        seed=42
    ).map(lambda x, y: (data_augmentation(x, training=True) / 255.0, y)).take(10)

    val_data = tf.keras.utils.image_dataset_from_directory(
        BASE_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        validation_split=0.2,
        subset='validation',
        seed=42
    ).map(lambda x, y: (x / 255.0, y)).take(3)


    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    history = model.fit(train_data, validation_data=val_data, epochs=TUNING_EPOCHS, callbacks=[early_stop], verbose=1)
    
    val_acc = max(history.history['val_accuracy'])
    print(f"✅ Batch {batch_size} → Max val accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_batch = batch_size

print(f"\n🏆 Best batch size selected: {best_batch}")



#FINAL TRAINING WITH BEST BATCH
train_data = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR,
    labels='inferred',
    label_mode='categorical',
    batch_size=best_batch,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    validation_split=0.2,
    subset='training',
    seed=42
)

val_data = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR,
    labels='inferred',
    label_mode='categorical',
    batch_size=best_batch,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    validation_split=0.2,
    subset='validation',
    seed=42
)

train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True) / 255.0, y))
val_data = val_data.map(lambda x, y: (x / 255.0, y))

#Modello CNN, tre layers
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# === PLOT ===
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model C - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#save models and errors 
model.save(MODEL_SAVE_PATH)

if os.path.exists(WRONG_DIR):
    shutil.rmtree(WRONG_DIR)
os.makedirs(os.path.join(WRONG_DIR, "images"), exist_ok=True)

rows = []
counter = 1
for image, label in val_data.unbatch():
    img_array = image.numpy()
    true_idx = int(np.argmax(label.numpy()))

    pred = model.predict(img_array[np.newaxis, ...], verbose=0)[0]
    pred_idx = int(np.argmax(pred))

    if pred_idx != true_idx:
        filename = f"wrong_{counter:03d}.jpg"
        path = os.path.join(WRONG_DIR, "images", filename)
        img_uint8 = (img_array * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(path)

        rows.append([filename, CLASS_NAMES[true_idx], CLASS_NAMES[pred_idx]])
        counter += 1

with open(os.path.join(WRONG_DIR, "wrong_preds.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "true_label", "predicted_label"])
    writer.writerows(rows)

print(f"\n✅ Model C saved to {MODEL_SAVE_PATH}")
print(f"📂 Wrong predictions saved to {WRONG_DIR}")
