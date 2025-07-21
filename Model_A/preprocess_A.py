import os
import cv2
import numpy as np
import random
import shutil


input_base  = r"D:/DESKTOP/Desktop/ML-project/rps-cv-images"          # origine
output_base = r"D:/DESKTOP/Desktop/ML-project/Model_A/rps-augmented"   #destin.
target_size = (300, 200)  # (w, h)

#check
if os.path.exists(output_base):
    shutil.rmtree(output_base)

#data augmentation - funzioni
def random_rotate(img, max_angle=25, min_angle=5):
    angle = random.choice([
        random.uniform(-max_angle, -min_angle),
        random.uniform(min_angle,  max_angle)
    ])
    M = cv2.getRotationMatrix2D(
        (img.shape[1] // 2, img.shape[0] // 2), angle, 1.0
    )
    return cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0]),
        borderMode=cv2.BORDER_REFLECT_101
    )

def random_brightness_contrast(img):
    alpha = random.uniform(0.6, 1.6)     #contrasto
    beta  = random.randint(-60, 60)      #luminosità
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def transform(img):
    #50 % flip orizzontale
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    #50 % rotazione
    if random.random() < 0.5:
        img = random_rotate(img)

    #luminosità/contrasto
    img = random_brightness_contrast(img)
    return img

#LOOP SU TUTTE LE IMMAGINi
for label in ['rock', 'paper', 'scissors']:
    src_dir  = os.path.join(input_base, label)
    dst_dir  = os.path.join(output_base, label)
    os.makedirs(dst_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img = cv2.imread(os.path.join(src_dir, fname))
        if img is None:
            continue

        img = cv2.resize(img, target_size)     
        img = transform(img)                    

        cv2.imwrite(os.path.join(dst_dir, fname), img)  #sovrascrivo non creo nuove

print("✅ Dataset A sovrascritto (stessa quantità di file) in:", output_base)
