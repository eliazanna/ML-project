import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

MODEL_PATH = "D:/DESKTOP/Desktop/ML-project/predictor_B.keras" #path predittore
IMG_W, IMG_H = 300, 200
CLASS_NAMES = ["paper", "rock", "scissors"]
model = tf.keras.models.load_model(MODEL_PATH)  #carico modello

#avvio media pipe 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw  = mp.solutions.drawing_utils

#avvio webcam
cap = cv2.VideoCapture(0)
print("Premi ESC per uscire")

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    #Mediapipe richiede RGB! mentre opencv restituisce in bgr.... 
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)

    label_text   = "Nessuna mano"
    preview_rgb  = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    if results.multi_hand_landmarks:
        h, w, _      = frame_rgb.shape
        lm            = results.multi_hand_landmarks[0].landmark
        x_min = int(min(pt.x for pt in lm) * w)
        y_min = int(min(pt.y for pt in lm) * h)
        x_max = int(max(pt.x for pt in lm) * w)
        y_max = int(max(pt.y for pt in lm) * h)

        #margine dinamico = 20 % lato lungo
        margin = int(max(x_max - x_min, y_max - y_min) * 0.2)
        x1, y1 = max(x_min - margin, 0), max(y_min - margin, 0)
        x2, y2 = min(x_max + margin, w), min(y_max + margin, h)

        #definisco le coordinate del box da ritagliare su cui predirre
        hand_bgr = frame_bgr[y1:y2, x1:x2] 
        if hand_bgr.size:                                        #evita frame vuoti
            hand_bgr = cv2.resize(hand_bgr, (IMG_W, IMG_H))
            hand_rgb = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2RGB)
            #hand_rgb = enhance_rgb(hand_rgb)

            #normalizzazione in 0-1 e batch
            inp = hand_rgb.astype(np.float32) / 255.0
            pred = model.predict(inp[np.newaxis, ...], verbose=0)[0]
            idx  = int(np.argmax(pred))
            conf = float(pred[idx])

            if conf < 0.40:
                label_text = "Not sure"
            else:
                label_text = f"{CLASS_NAMES[idx]} ({conf*100:.0f}%)"

            preview_rgb = hand_rgb.copy()

        #disegni a schermo di debug
        mp_draw.draw_landmarks(frame_bgr, results.multi_hand_landmarks[0],
                               mp_hands.HAND_CONNECTIONS)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #finestre a schermo
    cv2.putText(frame_bgr, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam", frame_bgr)
    cv2.imshow("Input al modello (RGB)", preview_rgb)

    if cv2.waitKey(1) & 0xFF == 27:      # ESC per uscire
        break