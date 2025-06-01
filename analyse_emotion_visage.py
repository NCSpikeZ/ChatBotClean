import cv2
from deepface import DeepFace

print("🟢 Démarrage de la webcam... Appuie sur Q pour quitter.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Impossible d’ouvrir la webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Problème de lecture du flux.")
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        print(f"💬 Émotion détectée : {emotion}")
        cv2.putText(frame, f"Émotion: {emotion}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    except Exception as e:
        print("⚠️ Erreur d'analyse :", e)

    cv2.imshow("Détection d'émotion", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
