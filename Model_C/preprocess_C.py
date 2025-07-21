import os
import cv2
import numpy as np
import random
import shutil

input_base = "D:/DESKTOP/Desktop/ML-project/rps-cv-images"
output_base = "D:/DESKTOP/Desktop/ML-project/Model_C/rps-processed"
target_size = (300, 200)

#check
if os.path.exists(output_base):
    shutil.rmtree(output_base)

#RANGE VERDE (HSV)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

#RIMOZIONE SFONDO VERDE + SFONDO BIANCO
def remove_green_and_add_white_bg(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    white_bg = np.full_like(img, 255)  # bianco puro
    result = img.copy()
    result[mask != 0] = white_bg[mask != 0]
    return result

#AUGMENTATION FORTE 
def apply_strong_augmentation(img):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    angle = random.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
    alpha = random.uniform(0.5, 1.7)  # contrasto
    beta  = random.randint(-70, 70)  # luminosità
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

#PROCESSA IMMAGINI
for label in ['rock', 'paper', 'scissors']:
    input_dir = os.path.join(input_base, label)
    output_dir = os.path.join(output_base, label)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(input_dir, filename)
            img = cv2.imread(path)
            if img is None:
                continue

            img = cv2.resize(img, target_size)
            img = remove_green_and_add_white_bg(img)
            img = apply_strong_augmentation(img)

            cv2.imwrite(os.path.join(output_dir, filename), img)

print(f"✅ Dataset Model C salvato in: {output_base}")
