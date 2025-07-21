import os
import cv2
import numpy as np
import random
import shutil
input_base = "D:/DESKTOP/Desktop/ML-project/rps-cv-images"
output_base = "D:/DESKTOP/Desktop/ML-project/rps-clean"
target_size = (300, 200)

#verifica su esistenza cartella
if os.path.exists(output_base):
    shutil.rmtree(output_base)

#verde
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

def get_random_bg_color():
    return np.random.randint(0, 256, size=3).tolist()

def remove_and_replace_bg(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    bg_color = get_random_bg_color()
    bg = np.full_like(img, bg_color)
    img[mask != 0] = bg[mask != 0]
    return img

def apply_random_augmentation(img):
    aug_type = random.choice(['flip', 'rotate', 'light'])

    if aug_type == 'flip':
        return cv2.flip(img, 1)
    elif aug_type == 'rotate':
        angle = random.choice([-25, -10, 10, 25])
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    elif aug_type == 'light':
        alpha = random.uniform(0.4, 1.6)
        beta = random.randint(-70, 70)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img

#ciclo immagini totali
for label in ['rock', 'paper', 'scissors']:
    input_dir = os.path.join(input_base, label)
    output_dir = os.path.join(output_base, label)
    os.makedirs(output_dir, exist_ok=True)

#applico augmentation random e cambio sfondo, salvo in cartella "clean"
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(input_dir, filename)
            img = cv2.imread(path)
            if img is None:
                continue

            img = remove_and_replace_bg(img)
            img = apply_random_augmentation(img)
            img = cv2.resize(img, target_size)

            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, img)