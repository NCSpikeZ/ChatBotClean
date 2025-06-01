import streamlit as st          # Interface web
import cv2                      # Traitement vidéo pour webcam
from fer import FER            # Reconnaissance d'émotions faciales
from transformers import pipeline  # Modèles d'IA pré-entraînés
import logging                 # Journalisation des erreurs
import numpy as np            # Calculs numériques
import requests               # Appels d'API
from datetime import datetime # Gestion des dates/heures
import uuid                   # Génération d'identifiants uniques

# Configuration de la page Streamlit
st.set_page_config(page_title="Modern Emotion AI", page_icon="🎭", layout="wide")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionnaire contenant les paramètres de chaque API disponible
API_CONFIGS = {
    "openai": {
        "name": "OpenAI GPT-4",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
        "requires_key": True
    },
    "groq": {
        "name": "Groq Llama 3.1",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.1-70b-versatile",
        "requires_key": True
    },
    "ollama": {
        "name": "Ollama Local",
        "endpoint": "http://localhost:11434/api/generate",
        "model": "llama3.1:8b",
        "requires_key": False
    },
}


# CHARGEMENT DES MODÈLES D'IA

@st.cache_resource
def load_text_classifier():
    """
    Charge le modèle de classification d'émotions pour le texte.
    Le cache évite de recharger le modèle à chaque interaction.
    """
    try:
        classifier = pipeline(
            "text-classification", 
            model="bhadresh-savani/distilbert-base-uncased-emotion", 
            top_k=1
        )
        logger.info("Text emotion classifier loaded successfully")
        return classifier
    except Exception as e:
        logger.error(f"Error loading classifier: {e}")
        return None

@st.cache_resource
def load_modern_text_generator():
    """
    Charge un générateur de texte conversationnel (fallback).
    Utilisé si les API externes ne fonctionnent pas.
    """
    try:
        generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            tokenizer="microsoft/DialoGPT-medium",
            device=-1
        )
        logger.info("Modern generator loaded successfully")
        return generator
    except Exception as e:
        logger.error(f"Error loading generator: {e}")
        return None


# CRÉATION DES PROMPTS ÉMOTIONNELS ADAPTATIFS

def create_emotional_prompt(emotion, original_text=None, api_type="openai", conversation_history=None):
    """
    Crée un prompt personnalisé selon l'émotion détectée.
    Adapte le format selon l'API utilisée (OpenAI/Groq vs Ollama).
    """
    base_context = f"You are an empathetic AI assistant. The user expresses a '{emotion}' emotion"
    if original_text:
        base_context += f" through the message: '{original_text}'"
    
    emotion_contexts = {
        "joy": "Share this joy and encourage this positive emotion.",
        "happy": "Celebrate this happiness and amplify this positive energy.",
        "sadness": "Offer comfort and hope with empathy.",
        "sad": "Be present and understanding, offer support.",
        "anger": "Help channel this anger constructively.",
        "angry": "Acknowledge this frustration and guide towards solutions.",
        "fear": "Reassure and help overcome this worry.",
        "scared": "Bring courage and coping strategies.",
        "surprise": "Share this amazement and explore this discovery.",
        "surprised": "Amplify this positive surprise.",
        "disgust": "Understand this rejection and help process these feelings.",
        "neutral": "Engage in natural and benevolent conversation.",
        "contempt": "Help transform this contempt into understanding."
    }
    
    context = emotion_contexts.get(emotion.lower(), "Respond with empathy and understanding.")
    
    # Format pour OpenAI et Groq (format messages)
    if api_type in ["openai", "groq"]:
        messages = [{"role": "system", "content": f"{base_context}. {context}"}]
        
        # Ajout de l'historique de conversation
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append(msg)
        
        messages.append({"role": "user", "content": original_text if original_text else f"I feel {emotion}"})
        return messages
        
    # Format pour Ollama
    elif api_type == "ollama":
        prompt = f"{base_context}. {context}\n\n"
        
        if conversation_history:
            prompt += "Previous conversation:\n"
            for msg in conversation_history[-6:]:
                role = "Human" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "\n"
        
        prompt += f"Current message: {original_text if original_text else f'I feel {emotion}'}\nResponse:"
        return prompt
    else:
        return original_text if original_text else f"I feel {emotion}"


# FONCTIONS D'APPEL DES API EXTERNES

def call_openai_api(messages, api_key, model="gpt-4o-mini"):
    """
    Appelle l'API OpenAI avec gestion d'erreurs complète.
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip(), "Success"
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return None, f"OpenAI Error: {str(e)}"

def call_groq_api(messages, api_key, model="llama-3.1-70b-versatile"):
    """
    Appelle l'API Groq (interface compatible OpenAI).
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip(), "Success"
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return None, f"Groq Error: {str(e)}"

def call_ollama_api(prompt, model="llama3.1:8b"):
    """
    Appelle une instance locale d'Ollama.
    Ollama doit être installé et en fonctionnement sur la machine.
    """
    try:
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 300
            }
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["response"].strip(), "Success"
        else:
            return None, f"Ollama Error: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return None, "Ollama not accessible. Make sure Ollama is running."
    except Exception as e:
        return None, f"Ollama Error: {str(e)}"


# FALLBACK

def get_enhanced_fallback_response(emotion, original_text=None):
    """
    Génère des réponses empathiques prédéfinies si les API externes échouent.
    Contient des réponses contextuelles pour chaque émotion.
    """
    # Détection de questions sur l'identité de l'IA
    if original_text and any(phrase in original_text.lower() for phrase in ["who are you", "what are you"]):
        return "I'm an AI assistant specialized in emotional analysis and empathetic conversations. How can I help you?"
    
    # Banque de réponses par émotion
    enhanced_responses = {
        "joy": [
            f"Your joy about '{original_text}' is truly contagious! It's wonderful to see someone radiating like this.",
            "This joy you express brightens our exchange! What makes you so happy today?",
            "I feel your happiness through your words. Keep cultivating this beautiful positive energy!"
        ] if original_text else [
            "Your joy is contagious! It's so beautiful to see someone radiating with happiness.",
            "This positive energy you emanate is truly inspiring!",
            "What a beautiful emotion! Keep cultivating this joy, it brightens your day."
        ],
        
        "sadness": [
            f"I understand that '{original_text}' can affect you deeply. It's important to welcome these emotions.",
            "Your feelings are legitimate and it's brave to express them. You're not alone in this trial.",
            "Sadness is part of the human experience. Take the time you need to get through this moment."
        ] if original_text else [
            "I understand your sadness. It's important to allow ourselves to feel these difficult emotions.",
            "Your feelings are valid. Sometimes expressing sadness is the first step toward healing.",
            "I'm here to listen. The sadness will pass, even if it seems overwhelming now."
        ],
        
        "anger": [
            f"I sense your frustration about '{original_text}'. This anger reveals something important to you.",
            "Your anger is understandable. How can we transform this energy into constructive action?",
            "This intense emotion deserves to be heard. What could help you feel better?"
        ] if original_text else [
            "I understand your anger. This powerful emotion often signals something important.",
            "Your frustration is legitimate. Let's take time to understand what's troubling you so much.",
            "Anger can be transformed into positive force. What would you like to change in this situation?"
        ],
        
        "fear": [
            f"Your concerns about '{original_text}' are understandable. Facing fears takes courage.",
            "This apprehension you feel is natural. You're stronger than you think.",
            "Your fears are heard. Together, we can find ways to tame them."
        ] if original_text else [
            "I understand your fears. It takes courage to recognize and express your fears.",
            "This worry is natural. You have the strength needed to get through this period.",
            "Your fears are legitimate. Let's take them one by one, at your pace."
        ],
        
        "neutral": [
            f"Thank you for sharing '{original_text}' with me. How can I better accompany you today?",
            "Your message interests me. Would you like to explore this topic more deeply?",
            "I'm here to exchange with you. What's concerning or interesting you right now?"
        ] if original_text else [
            "I'm here to listen and accompany you. What would you like to talk about today?",
            "Your presence is appreciated. How can I be useful in our exchanges?",
            "Let's take time for an authentic conversation. What's close to your heart?"
        ]
    }
    
    # Sélection aléatoire
    import random
    emotion_clean = emotion.lower().strip()
    
    if emotion_clean in enhanced_responses:
        responses = enhanced_responses[emotion_clean]
        return random.choice(responses)
    else:
        if original_text:
            return f"I understand your message '{original_text}' and the emotion that accompanies it. How can I help you feel better?"
        else:
            return f"I perceive that you feel {emotion}. Your emotions are important and deserve to be heard."

# Chargement des modèles au démarrage de l'application
text_classifier = load_text_classifier()
modern_generator = load_modern_text_generator()

# Variables de session Streamlit (persistent pendant la session utilisateur)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Historique complet des conversations

if 'conversation_messages' not in st.session_state:
    st.session_state.conversation_messages = []  # Messages pour le contexte IA

# Variables pour la gestion de la webcam
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'webcam_emotion_result' not in st.session_state:
    st.session_state.webcam_emotion_result = {"emotion": "neutral", "score": 0.0}
if 'webcam_capture' not in st.session_state:
    st.session_state.webcam_capture = None
if 'webcam_detector' not in st.session_state:
    st.session_state.webcam_detector = None


# FONCTIONS DE GESTION DE LA WEBCAM

def initialize_webcam():
    """
    Initialise la webcam en testant plusieurs indices de caméra.
    Retourne True si une caméra fonctionnelle est trouvée.
    """
    try:
        for i in range(3):  # Teste les 3 premiers indices de caméra
            cap = cv2.VideoCapture(i)
            ret, _ = cap.read()
            if ret:
                st.session_state.webcam_capture = cap
                st.session_state.webcam_detector = FER(mtcnn=False)
                logger.info(f"Camera initialized at index {i}")
                return True
            cap.release()
        return False
    except Exception as e:
        logger.error(f"Webcam initialization error: {e}")
        return False

def capture_webcam_emotion():
    """
    Capture une image de la webcam et détecte l'émotion du visage.
    Retourne l'émotion dominante et son score de confiance.
    """
    if not st.session_state.webcam_capture or not st.session_state.webcam_detector:
        return {"emotion": "neutral", "score": 0.0}, "Webcam not initialized"
    
    try:
        ret, frame = st.session_state.webcam_capture.read()
        if not ret:
            return {"emotion": "neutral", "score": 0.0}, "Capture failed"
        
        # Redimensionne l'image pour améliorer les performances
        frame = cv2.resize(frame, (640, 480))
        result = st.session_state.webcam_detector.detect_emotions(frame)
        
        if result and len(result) > 0:
            face_emotions = result[0]['emotions']
            dominant_emotion = max(face_emotions, key=face_emotions.get)
            confidence = face_emotions[dominant_emotion]
            return {"emotion": dominant_emotion, "score": confidence}, "Success"
        else:
            return {"emotion": "neutral", "score": 0.0}, "No face detected"
            
    except Exception as e:
        return {"emotion": "neutral", "score": 0.0}, f"Detection error: {str(e)}"

def cleanup_webcam():
    """
    Nettoie les ressources de la webcam quand elle n'est plus utilisée.
    """
    if st.session_state.webcam_capture:
        st.session_state.webcam_capture.release()
        st.session_state.webcam_capture = None
    st.session_state.webcam_detector = None


# GESTION DE L'HISTORIQUE DES CONVERSATIONS

def add_to_chat_history(user_message, ai_response, emotion, confidence, model_used):
    """
    Ajoute une nouvelle entrée à l'historique des conversations.
    Maintient aussi le contexte pour les API d'IA.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    chat_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp,
        "user_message": user_message,
        "ai_response": ai_response,
        "emotion": emotion,
        "confidence": confidence,
        "model_used": model_used
    }
    st.session_state.chat_history.append(chat_entry)
    
    # Mise à jour du contexte pour les IA
    st.session_state.conversation_messages.append({"role": "user", "content": user_message})
    st.session_state.conversation_messages.append({"role": "assistant", "content": ai_response})
    
    if len(st.session_state.conversation_messages) > 20:
        st.session_state.conversation_messages = st.session_state.conversation_messages[-20:]

def display_chat_history():
    """
    Affiche l'historique des conversations dans l'interface utilisateur.
    """
    if not st.session_state.chat_history:
        st.info("💬 No conversation yet. Start chatting!")
        return
    
    st.markdown("### 💬 Conversation History")
    
    with st.container():
        # Affiche les 10 dernières conversations (ordre inversé)
        for i, entry in enumerate(reversed(st.session_state.chat_history[-10:])):
            with st.expander(f"💬 {entry['timestamp']} - {entry['emotion'].title()} ({entry['confidence']*100:.0f}%)", expanded=(i==0)):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**👤 You:**")
                    st.markdown(f"_{entry['user_message']}_")
                
                with col2:
                    st.markdown(f"**🤖 AI ({entry['model_used']}):**")
                    st.markdown(entry['ai_response'])

# Variable pour gérer l'état de génération
if 'generation_state' not in st.session_state:
    st.session_state.generation_state = {
        'show_process': False,
        'steps': [],
        'final_result': None,
        'error': None,
        'processing': False
    }

# En-tête de l'application
st.title("🎭 Emotion AI with Continuous Chat")
st.markdown("**Analyze emotions and have continuous conversations with empathetic AI**")


# 12. BARRE LATÉRALE

st.sidebar.header("🔧 API Configuration")
selected_api = st.sidebar.selectbox(
    "Choose AI API:",
    options=list(API_CONFIGS.keys()),
    format_func=lambda x: API_CONFIGS[x]["name"]
)

# Gestion des clés API
api_key = None
if API_CONFIGS[selected_api]["requires_key"]:
    api_key = st.sidebar.text_input(
        f"{API_CONFIGS[selected_api]['name']} API Key:",
        type="password"
    )
    
    if not api_key and selected_api != "huggingface":
        st.sidebar.warning(f"⚠️ API key required for {API_CONFIGS[selected_api]['name']}")

# Guide pour obtenir les clés API
with st.sidebar.expander("📋 How to get API keys"):
    st.write("""
    **OpenAI**: https://platform.openai.com/api-keys
    **Groq**: https://console.groq.com/keys
    **Ollama**: Local installation (free)
    """)


# FONCTION DE PRÉVISUALISATION D'ÉMOTION

def detect_emotion_preview(text, emotion_source="Auto-detect"):
    """
    Détecte l'émotion pour l'aperçu en temps réel pendant que l'utilisateur tape.
    """
    if not text.strip():
        return None, 0.0
    
    if emotion_source == "Auto-detect" and text_classifier:
        result = text_classifier(text)[0]
        return result[0]['label'], result[0]['score']
    elif emotion_source == "Webcam":
        return st.session_state.webcam_emotion_result["emotion"], st.session_state.webcam_emotion_result["score"]
    
    return "neutral", 0.5


st.sidebar.markdown("---")
st.sidebar.header("💬 Chat Management")
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.conversation_messages = []
    st.success("Chat history cleared!")
    st.rerun()

st.sidebar.metric("Messages", len(st.session_state.chat_history))


# INTERFACE DE CHAT PRINCIPAL

st.subheader("💬 Continuous Chat Interface")

chat_col1, chat_col2 = st.columns([3, 1])

with chat_col1:
    user_message = st.text_input(
        "💭 Type your message:",
        placeholder="How are you feeling today? Tell me what's on your mind...",
        key="chat_input"
    )

with chat_col2:
    emotion_source = st.selectbox(
        "Emotion source:",
        ["Auto-detect", "Webcam"],
        index=0
    )

# Prévisualisation de l'émotion en temps réel
if user_message.strip():
    preview_emotion, preview_score = detect_emotion_preview(user_message, emotion_source)
    if preview_emotion:
        emotion_color = {
            'joy': '🟢', 'happy': '🟢',
            'sadness': '🔵', 'sad': '🔵', 
            'anger': '🔴', 'angry': '🔴',
            'fear': '🟡', 'scared': '🟡',
            'surprise': '🟠', 'surprised': '🟠',
            'neutral': '⚪'
        }
        
        color_indicator = emotion_color.get(preview_emotion, '⚪')
        st.info(f"{color_indicator} Emotion detected: **{preview_emotion.title()}** ({preview_score*100:.0f}% confidence)")

# Bouton d'envoi centré
send_col1, send_col2, send_col3 = st.columns([1, 1, 1])

with send_col2:
    send_message = st.button("📤 Send Message", type="primary", use_container_width=True)


# TRAITEMENT DU MESSAGE ET GÉNÉRATION DE RÉPONSE

if send_message and user_message.strip():
    with st.spinner("🤖 AI is thinking..."):
        try:
            # Détection de l'émotion selon la source choisie
            if emotion_source == "Auto-detect":
                if text_classifier:
                    text_result = text_classifier(user_message)[0]
                    emotion_to_use = text_result[0]['label']
                    emotion_score = text_result[0]['score']
                else:
                    emotion_to_use = "neutral"
                    emotion_score = 0.5
            else:  # Webcam
                emotion_to_use = st.session_state.webcam_emotion_result["emotion"]
                emotion_score = st.session_state.webcam_emotion_result["score"]
            
            # Création du prompt adaptatif
            prompt = create_emotional_prompt(
                emotion_to_use, 
                user_message, 
                selected_api, 
                st.session_state.conversation_messages
            )
            
            # Appel de l'API sélectionnée
            response_text = None
            model_used = "None"
            
            if selected_api == "openai" and api_key:
                response_text, status = call_openai_api(prompt, api_key)
                model_used = "OpenAI GPT-4"
            elif selected_api == "groq" and api_key:
                response_text, status = call_groq_api(prompt, api_key)
                model_used = "Groq Llama 3.1"
            elif selected_api == "ollama":
                response_text, status = call_ollama_api(prompt)
                model_used = "Ollama Local"
            else:
                status = "API not configured or missing key"
            
            # Système de fallback si l'API échoue
            if not response_text:
                response_text = get_enhanced_fallback_response(emotion_to_use, user_message)
                model_used = "Enhanced Fallback"
            
            # Ajout à l'historique et rafraîchissement
            add_to_chat_history(user_message, response_text, emotion_to_use, emotion_score, model_used)
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")

# Affichage de l'historique des conversations
display_chat_history()


# 17. SECTION WEBCAM

with st.expander("Webcam Emotion Detection", expanded=False):
    webcam_col1, webcam_col2, webcam_col3 = st.columns(3)
    
    # Boutons de contrôle de la webcam
    with webcam_col1:
        if st.button("🔴 Start" if not st.session_state.webcam_active else "⏹️ Stop"):
            if not st.session_state.webcam_active:
                if initialize_webcam():
                    st.session_state.webcam_active = True
                    st.success("Webcam started!")
                else:
                    st.error("Webcam initialization failed")
            else:
                cleanup_webcam()
                st.session_state.webcam_active = False
                st.info("Webcam stopped")
            st.rerun()
    
    with webcam_col2:
        if st.button("📸 Capture", disabled=not st.session_state.webcam_active):
            if st.session_state.webcam_active:
                emotion_result, status = capture_webcam_emotion()
                st.session_state.webcam_emotion_result = emotion_result
                
                if status == "Success":
                    st.success(f"Captured: {emotion_result['emotion'].title()} ({emotion_result['score']*100:.1f}%)")
                else:
                    st.warning(f"Status: {status}")
                st.rerun()
    
    with webcam_col3:
        if st.button("🔄 Clear", disabled=not st.session_state.webcam_active):
            st.session_state.webcam_emotion_result = {"emotion": "neutral", "score": 0.0}
            st.rerun()
    
    # Statut et métriques de la webcam
    if st.session_state.webcam_active:
        st.info("🟢 Webcam active")
    else:
        st.info("🔴 Webcam inactive")
    
    current_emotion = st.session_state.webcam_emotion_result["emotion"]
    current_score = st.session_state.webcam_emotion_result["score"] * 100
    
    st.metric(
        label="Last Captured Emotion", 
        value=current_emotion.title(), 
        delta=f"{current_score:.1f}% confidence"
    )


# FONCTIONNALITÉS AVANCÉES

with st.expander("🔧 Advanced Features", expanded=False):
    
    # Analytiques des conversations
    st.subheader("📊 Conversation Analytics")
    if st.session_state.chat_history:
        emotions = [entry['emotion'] for entry in st.session_state.chat_history]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        st.write("**Emotion Distribution:**")
        for emotion, count in emotion_counts.items():
            st.write(f"- {emotion.title()}: {count} messages")
        
        # Confiance moyenne en la reconnaissance d'émotion
        avg_confidence = sum(entry['confidence'] for entry in st.session_state.chat_history) / len(st.session_state.chat_history)
        st.metric("Average Emotion Confidence", f"{avg_confidence*100:.1f}%")
    
    st.subheader("💾 Export Conversation")
    if st.button("📄 Export as Text"):
        if st.session_state.chat_history:
            export_text = "Conversation Export\n" + "="*50 + "\n\n"
            for entry in st.session_state.chat_history:
                export_text += f"[{entry['timestamp']}] Emotion: {entry['emotion']} ({entry['confidence']*100:.0f}%)\n"
                export_text += f"You: {entry['user_message']}\n"
                export_text += f"AI ({entry['model_used']}): {entry['ai_response']}\n\n"
            
            st.download_button(
                label="📥 Download Conversation",
                data=export_text,
                file_name=f"emotion_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.info("No conversation to export")

st.markdown("---")
st.markdown("""
**📋 Usage Guide:**

1. **Setup:** 
   - Choose your preferred API in the sidebar
   - Add your API key if required
   - Ollama runs locally and doesn't need a key

2. **Continuous Chat:** 
   - Type your message in the text input
   - Choose emotion detection method (Auto-detect or Webcam)
   - Click "Send Message" to chat with the AI
   - The AI remembers your conversation context

3. **Features:** 
   - Continuous conversation with context memory
   - Optional webcam emotion detection
   - Conversation analytics and export
   - Emotion-aware AI responses

**🆕 New Chat Features:**
- Continuous conversation with memory
- Context-aware responses
- Chat history with timestamps
- Conversation analytics
- Export functionality
""")

with st.sidebar:
    st.header("📊 System Status")
    
    st.write("**Components:**")
    st.write(f"- Classifier: {'✅' if text_classifier else '❌'}")
    st.write(f"- Webcam: {'🟢' if st.session_state.webcam_active else '🔴'}")
    st.write(f"- HF Generator: {'✅' if modern_generator else '❌'}")
    
    st.write("**Selected API:**")
    st.write(f"- {API_CONFIGS[selected_api]['name']}")
    if API_CONFIGS[selected_api]["requires_key"]:
        st.write(f"- Key: {'✅' if api_key else '❌'}")
    else:
        st.write("- Key: Not required")
    
    if st.session_state.chat_history:
        st.write("**Chat Stats:**")
        st.write(f"- Messages: {len(st.session_state.chat_history)}")
        st.write(f"- Context: {len(st.session_state.conversation_messages)} msgs")
    
    st.markdown("---")
    if st.button("Restart Application"):
        cleanup_webcam()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
        
    st.markdown("---")
    st.markdown("**🔗 Useful Links:**")
    st.markdown("- [OpenAI API](https://platform.openai.com/)")
    st.markdown("- [Groq](https://console.groq.com/)")
    st.markdown("- [Ollama](https://ollama.ai/)")