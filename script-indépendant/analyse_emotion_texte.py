from transformers import pipeline

# Modèle fiable en anglais
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

def detect_text_emotion(text):
    result = classifier(text)
    emotion = result[0][0]['label']   # Correction ici 👈
    score = result[0][0]['score']
    return emotion, round(score, 2)

while True:
    texte = input("📝 Enter a sentence (or type 'exit' to quit): ")
    if texte.lower() == "exit":
        break

    emotion, confidence = detect_text_emotion(texte)
    print(f"💬 Detected emotion: {emotion} ({confidence * 100:.1f} %)")
