# webcam_emotion_detector.py

import cv2
import json
import time
from fer import FER

def main():
    detector = FER(mtcnn=True)  # Utilise MTCNN pour une meilleure précision
    cap = cv2.VideoCapture(0)   # Ouvre la webcam

    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    print("✅ Webcam emotion detection started.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Détection de l’émotion
        emotions = detector.detect_emotions(frame)
        if emotions:
            top_emotion, score = detector.top_emotion(frame)
            # Sauvegarde dans un fichier JSON
            with open("emotion.json", "w") as f:
                json.dump({"emotion": top_emotion, "score": score}, f)
            print(f"🧠 Emotion: {top_emotion} ({score:.2f})")

        time.sleep(1)  # Pause de 1s pour éviter trop de traitements

if __name__ == "__main__":
    main()
