import gradio as gr
import sounddevice as sd
import numpy as np
import cv2
import time
import os
os.environ['GLOG_minloglevel'] = '2'  # Cache les logs info/debug de MediaPipe
import mediapipe as mp
from deepface import DeepFace
from transformers import AutoProcessor, AutoModelForAudioClassification
#from llama_cpp import Llama
import pyttsx3
import torch
import speech_recognition as sr
import soundfile as sf
from transformers import pipeline
import json 


print("Initialisation de MediaPipe FaceMesh...")
# Initialisation des modèles
#mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
 #   max_num_faces=1,
  #  refine_landmarks=True,
   # min_detection_confidence=0.7,
    #static_image_mode=False
#)
print("MediaPipe FaceMesh initialisé avec succès.")


"""llm = Llama(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    n_batch=512,
    verbose=False
)"""

# Chargez Wav2Vec2 avec la nouvelle API
try:
    audio_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
    audio_model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
except Exception as e:
    print(f"Erreur chargement Wav2Vec2: {e}")
    # Fallback
    audio_processor, audio_model = None, None

# Transcription avec Whisper
try:
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
except Exception as e:
    print(f"Erreur chargement Whisper: {e}")
    transcriber = None


"""
def transcribe_audio(audio_data):
    
    r = sr.Recognizer()
    try:
        # Sauvegarde temporaire du fichier audio
        with open("temp.wav", "wb") as f:
            sf.write(f, audio_data, 16000)
        
        with sr.AudioFile("temp.wav") as source:
            audio = r.record(source)
        return r.recognize_google(audio, language="fr-FR")
    except Exception as e:
        print(f"Erreur transcription: {e}")
        return ""
"""

import re

def extract_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError as e:
        print("❌ JSON invalide :", e)
    return {
        "score": 0,
        "points_forts": [],
        "points_faibles": [],
        "avis": "Erreur d'analyse"
    }


import requests
import json

def analyze_response_llama_local(question, transcription, job_title="Développeur"):
    prompt = f"""
Tu es un recruteur pour le poste de {job_title}. 
Tu ne dois parler qu’en français
Voici la question posée au candidat : {question}
Voici sa réponse transcrite : {transcription}

Évalue cette réponse et fournis une analyse sous format JSON avec les champs :
- score : une note entre 0 et 10
- points_forts : liste des points forts
- points_faibles : liste des points faibles
- avis : phrase d’évaluation finale

Format attendu :
{{
  "score": int,
  "points_forts": [...],
  "points_faibles": [...],
  "avis": "..."
}}
"""

    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })

    try:
        content = res.json()["response"]
        print("🔍 Réponse brute LLaMA :\n", content)
        return extract_json(content)

    except Exception as e:
        print("❌ Erreur parsing JSON :", e)
        return {
            "score": 0,
            "points_forts": [],
            "points_faibles": [],
            "avis": "Analyse échouée"
        }





def generate_question():
    prompt = """Tu es un intervieweur technique strict. Génère UNIQUEMENT une question technique en français 
sans aucun commentaire, contexte ou réponse. Format exigé : 'Question sur [sujet technique] ?' 

Voici des exemples valides :
- 'Comment optimiseriez-vous une requête SQL ?'
- 'Expliquez le principe du MVC ?'

Génère maintenant une question technique :"""

    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })

    question = res.json()["response"].strip()
    if not question.endswith('?'):
        question = question.split('?')[0] + '?'

    return question


def analyze_face(frame):
    try:
        with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            static_image_mode=False
        ) as mp_face_mesh:
            results = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = []
            if results.multi_face_landmarks:
                landmarks = [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]

            emotions = DeepFace.analyze(frame, actions=['emotion'], detector_backend='mtcnn')[0]
            
            return {
                "landmarks": len(landmarks),
                "emotion": emotions['dominant_emotion'],
                "confidence": float(emotions['emotion'][emotions['dominant_emotion']])
            }

    except Exception as e:
        print(f"Erreur analyse faciale: {e}")
        return "Analyse visuelle indisponible."



def analyze_audio(audio_data):
    if audio_processor is None:
        return {"emotion": "neutral", "confidence": 0.5}
    
    try:
        inputs = audio_processor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            outputs = audio_model(**inputs)
        return {
            "emotion": "happy" if outputs.logits[0][1] > 0 else "neutral",
            "confidence": float(outputs.logits.softmax(dim=1)[0][1])
        }
    except Exception as e:
        print(f"Erreur analyse audio: {e}")
        return {"error": str(e)}
    
def transcribe_audio(audio_data, return_timestamps=False):
    """Convertit l'audio en texte, debug prints inclus."""
    if transcriber is None:
        print("❌ Transcriber non initialisé.")
        return "Transcription non disponible"
    try:
        audio_np = np.array(audio_data, dtype=np.float32)
        print(f"🔍 [DEBUG] audio_np.shape = {audio_np.shape}, dtype = {audio_np.dtype}")
        duration_s = len(audio_np) / 16000
        print(f"🔍 [DEBUG] Durée de l'audio = {duration_s:.2f} s")
        if duration_s < 1.0:
            print("⚠️ [DEBUG] Audio trop court (<1 s) pour une bonne transcription.")
        
        # ⚠️ Retiré sampling_rate
        #result = transcriber(audio_np)
        result = transcriber(audio_np, generate_kwargs={"language": "fr"}, return_timestamps=return_timestamps)
        print(f"✅ [DEBUG] transcription brute = {result['text']}")
        return result["text"]
    except Exception as e:
        print(f"❌ Erreur transcription: {e}")
        return "Erreur lors de la transcription"



def analyze_response_text(response_text):
    prompt = f"""Voici une réponse donnée à une question d'entretien :
"{response_text}"
Analyse cette réponse en identifiant les points forts, les faiblesses, et donne un score sur 10.
Réponds sous forme d'objet JSON avec les champs :
"score", "points_forts", "points_faibles"."""

    try:
        res = llm(prompt, max_tokens=300, temperature=0.7)
        output_text = res['choices'][0]['text']
        return eval(output_text)
    except Exception as e:
        print(f"Erreur analyse réponse: {e}")
        return {
            "score": 0,
            "points_forts": [],
            "points_faibles": ["Analyse indisponible"]
        }

def process_response(audio_data, question, job_title="Développeur"):
    transcription = transcribe_audio(audio_data, return_timestamps=True)
    analysis = analyze_response_llama_local(question, transcription, job_title)
    return transcription, analysis



def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Erreur synthèse vocale: {e}")

# ... (vos imports restent identiques)
def flatten_list(lst):
    flat = []
    for item in lst:
        if isinstance(item, list):
            # Si c'est une sous-liste, on l'étale dans la liste plate
            flat.extend(item)
        else:
            flat.append(item)
    return flat



from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def export_to_pdf(analysis_data, file_name="resultat_entretien.pdf"):
    try:
        c = canvas.Canvas(file_name, pagesize=letter)
        width, height = letter

        # Titre
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, height - 40, "Résumé de l'Entretien Technique")

        # Résumé de la réponse
        y_position = height - 60
        c.setFont("Helvetica", 12)
        c.drawString(100, y_position, "Transcription de la Réponse:")
        y_position -= 20
        c.drawString(100, y_position, analysis_data["reponse"])

        # Points forts
        y_position -= 40
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, y_position, "Points Forts:")
        y_position -= 20
        for point in analysis_data["points_forts"]:
            c.setFont("Helvetica", 12)
            c.drawString(100, y_position, f"- {point}")
            y_position -= 20

        # Points faibles
        y_position -= 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, y_position, "Points Faibles:")
        y_position -= 20
        for point in analysis_data["points_faibles"]:
            c.setFont("Helvetica", 12)
            c.drawString(100, y_position, f"- {point}")
            y_position -= 20

        # Emotion détectée
        y_position -= 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, y_position, "Émotion détectée:")
        y_position -= 20
        c.setFont("Helvetica", 12)
        c.drawString(100, y_position, f"{analysis_data['emotion']} (Confiance: {analysis_data['confidence']}%)")

        # Finaliser le PDF
        c.save()
        print(f"Le fichier PDF a été généré avec succès : {file_name}")
        return file_name
    except Exception as e:
        print(f"Erreur lors de la génération du PDF : {e}")
        return None




    # LA FONCTION DOIT ÊTRE DÉFINIE AVANT DE L'UTILISER
def run_interview():
    try:
        # 🗣️ Message d’intro pour le candidat
        message_intro = (
            "Merci de répondre clairement à la question.\n"
            "Placez-vous près du micro et assurez-vous d’être dans un environnement calme.\n"
            "Vous avez 60 secondes pour répondre."
        )
        print(message_intro)
        speak(message_intro)
        yield None, "", message_intro, None, None

        # Génération question
        q = generate_question()
        speak(q)
        yield None, q, "Question posée (60s pour répondre)", None, None
        
        # Capture audio/vidéo
        audio_data = sd.rec(int(60 * 16000), samplerate=16000, channels=1)
        cap = cv2.VideoCapture(0)
        frames = []
        
        for i in range(60):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                yield frame, q, f"Temps restant: {59 - i}s", None, None
                time.sleep(1)
        
        # Analyses
        face_result = analyze_face(frames[-1])
        audio_result = analyze_audio(audio_data.flatten())
        
        # NOUVEAU: Analyse de la réponse
        transcription, content_analysis = process_response(audio_data.flatten(), q, "Développeur")
        
        # Nettoyage pour affichage lisible
        if isinstance(face_result, dict):
            face_summary = (
                f"Émotion détectée: {face_result['emotion']} "
                f"(Confiance: {round(face_result['confidence']*100)}%)\n"
                f"Points détectés: {face_result['landmarks']}"
            )
        else:
            face_summary = face_result  # chaîne "Analyse visuelle indisponible."

        audio_summary = f"Émotion détectée: {audio_result.get('emotion', 'inconnue')} (Confiance: {round(audio_result.get('confidence', 0)*100)}%)"
        # Aplatir les points forts et points faibles si besoin
        pf = flatten_list(content_analysis.get('points_forts', []))
        pfa = flatten_list(content_analysis.get('points_faibles', []))

        reponse_summary = f"Transcription: {transcription.strip()}\n"
        if content_analysis and isinstance(content_analysis, dict):
            reponse_summary += f"Score: {content_analysis.get('score', 0)}\n"
            reponse_summary += f"Points forts: {', '.join(pf) or 'Aucun'}\n"
            reponse_summary += f"Points faibles: {', '.join(pfa) or 'Aucun'}"
        else:
            reponse_summary += "Analyse du contenu indisponible."


        # Tout ce qu'on affiche à la fin
        resume_json = {
    "audio": audio_summary,
    "face": face_summary,
    "reponse": reponse_summary,
    "points_forts": pf,
            "points_faibles": pfa,
            "emotion": audio_result.get('emotion', 'inconnue'),
            "confidence": round(audio_result.get('confidence', 0) * 100)
}
        # Exporter les résultats en PDF
        pdf_file = export_to_pdf(resume_json)

        yield frames[-1], q, "Analyse terminée", resume_json, transcription, pdf_file



        
    except Exception as e:
        print(f"Erreur globale: {e}")
        yield None, "Erreur", str(e), None, None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🤖 Interview Technique")
    
    with gr.Row():
        webcam = gr.Image(label="Webcam Live", width=640)
        with gr.Column():
            question = gr.Textbox(label="Question", lines=3)
            with gr.Accordion("Résultats détaillés", open=False):
                results = gr.JSON(label="Analyse complète")
                transcription_box = gr.Textbox(label="Transcription", interactive=False)
                pdf_output = gr.File(label="Télécharger le PDF")
            status = gr.Textbox(label="Statut")        

        gr.Button("Démarrer l'interview").click(
    run_interview,
    outputs=[webcam, question, status, results, transcription_box]
)




demo.launch(share=False)