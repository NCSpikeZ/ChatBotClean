from transformers import pipeline

# Load English emotion detection model
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Basic bot responses based on emotion
emotion_responses = {
    "anger": "I'm here if you want to talk about it. Let's try to stay calm together.",
    "disgust": "That sounds unpleasant. I'm sorry you're feeling that way.",
    "fear": "You’re not alone. I'm here with you.",
    "joy": "I'm glad to hear that! 😊",
    "neutral": "Thanks for sharing. I'm listening.",
    "sadness": "I'm sorry you're feeling down. Want to talk about it?",
    "surprise": "Oh wow! That sounds unexpected!"
}

def detect_emotion(text):
    result = classifier(text)
    label = result[0][0]['label']
    score = result[0][0]['score']
    return label, round(score, 2)

def generate_response(emotion):
    return emotion_responses.get(emotion.lower(), "I'm here to support you.")

if __name__ == "__main__":
    print("👋 Hi! I'm your emotional support bot.")
    while True:
        text = input("📝 Tell me something (or type 'exit' to quit): ")
        if text.lower() == "exit":
            print("👋 Take care! See you next time.")
            break

        emotion, confidence = detect_emotion(text)
        response = generate_response(emotion)

        print(f"🔍 Detected emotion: {emotion} ({confidence * 100:.1f}%)")
        print(f"🤖 Bot: {response}")
