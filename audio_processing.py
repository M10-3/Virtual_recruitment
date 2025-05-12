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
    prompt = """Tu es un intervieweur bienveillant mais exigeant. Génére uniquement UNE question pertinente en français, 
en lien avec le développement logiciel, pouvant être :
- plus générale (comme l’organisation du travail, la communication technique, la veille technologique, etc.)

Ne donne que la question, sans commentaire ni explication.
Format : une phrase interrogative en français, terminée par un point d'interrogation.

Exemples :
- "Décrivez votre dernier emploi ?"
- "Pourquoi souhaitez-vous travailler ici ?"
- "Comment avez-vous traité un conflit au travail ?"
- "Comment travaillez-vous en équipe sur un projet technique ?"
- "Comment réagissez-vous face à un bug difficile à identifier ?"

Génère maintenant une question :"""

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

def wrap_text(c, text, x, y, max_width, font="Helvetica", font_size=12, line_height=16):
    """Écrit le texte avec retour à la ligne automatique, retourne la nouvelle position y."""
    c.setFont(font, font_size)
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        if c.stringWidth(test_line, font, font_size) < max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    for line in lines:
        if y < 50:  # limite bas de page
            c.showPage()
            y = letter[1] - 50
            c.setFont(font, font_size)
        c.drawString(x, y, line)
        y -= line_height

    return y

def export_to_pdf(analysis_data, file_name="resultat_entretien.pdf"):
    try:
        c = canvas.Canvas(file_name, pagesize=letter)
        width, height = letter
        margin = 50
        y = height - margin

        # Titre
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "Résumé de l'Entretien Technique")
        y -= 30

        # Transcription de la réponse
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Transcription de la Réponse:")
        y -= 20
        y = wrap_text(c, analysis_data["reponse"], margin, y, width - 2 * margin)

        # Points forts
        y -= 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Points Forts:")
        y -= 20
        for point in analysis_data["points_forts"]:
            y = wrap_text(c, f"- {point}", margin, y, width - 2 * margin)

        # Points faibles
        y -= 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Points Faibles:")
        y -= 20
        for point in analysis_data["points_faibles"]:
            y = wrap_text(c, f"- {point}", margin, y, width - 2 * margin)

        # Emotion détectée
        y -= 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Émotion détectée:")
        y -= 20
        y = wrap_text(c, f"{analysis_data['emotion']} (Confiance: {analysis_data['confidence']}%)", margin, y, width - 2 * margin)

        # Finaliser
        c.save()
        print(f"✅ Fichier PDF généré avec succès : {file_name}")
        return file_name

    except Exception as e:
        print(f"❌ Erreur lors de la génération du PDF : {e}")
        return None




def run_interview():
    try:
        N_QUESTIONS = 1  # Tu peux ajuster ici le nombre de questions
        analyses = []
        all_audio = []
        all_frames = []
        all_questions = []

        message_intro = (
            f"Merci de répondre à {N_QUESTIONS} questions techniques.\n"
            "Prenez une grande inspiration, ça va bien se passer !"
        )
        speak(message_intro)
        yield None, "", message_intro, None, None, None, None, None, None, None

        for i in range(N_QUESTIONS):
            speak(f"Question {i+1}")
            q = generate_question()
            speak(q)
            yield None, q, f"Question {i+1} posée (60s pour répondre)", None, None, None, None, None, None, None

            # Enregistrement audio + webcam
            audio_data = sd.rec(int(60 * 16000), samplerate=16000, channels=1)
            cap = cv2.VideoCapture(0)
            frames = []
            for j in range(60):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    yield frame, q, f"Q{i+1} - Temps restant: {59-j}s", None, None, None, None, None, None, None
                    time.sleep(1)
            cap.release()

            all_audio.append(audio_data.flatten())
            all_frames.append(frames[-1])
            all_questions.append(q)

            if i < N_QUESTIONS - 1:
                speak("D'accord, merci pour votre réponse, on passe à la question suivante.")

            yield frames[-1], q, f"Q{i+1} terminée", None, None, None, None, None, None, None

        speak("Parfait ! Merci pour votre attention, l'entretien est terminé.")
        yield None, "", "Traitement des réponses en cours... ça peut prendre quelques instants", None, None, None, None, None, None, None

        # 🧠 Traitement des réponses après toutes les questions
        for i in range(N_QUESTIONS):
            transcription, content_analysis = process_response(all_audio[i], all_questions[i], "Développeur")
            face_result = analyze_face(all_frames[i])
            audio_result = analyze_audio(all_audio[i])
            # 🌈 Explication locale audio
            impacts, _ = explain_audio_local(all_audio[i], audio_model, audio_processor)
            plot_audio_explanation(all_audio[i], impacts, output_path=f"outputs/audio_xai_q{i+1}.png")

            xai_result = explain_response_with_lime(transcription)
            xai_result_face = explain_face_with_lime(all_frames[i]) 


            pf = flatten_list(content_analysis.get('points_forts', []))
            pfa = flatten_list(content_analysis.get('points_faibles', []))

            face_summary = (
                f"Émotion détectée: {face_result['emotion']} "
                f"(Confiance: {round(face_result['confidence'])}%)"
                if isinstance(face_result, dict)
                else face_result
            )
            audio_summary = f"Émotion détectée: {audio_result.get('emotion', 'inconnue')} (Confiance: {round(audio_result.get('confidence', 0)*100)}%)"

            reponse_summary = f"Q{i+1} - Transcription: {transcription.strip()}\n"
            reponse_summary += f"Score: {content_analysis.get('score', 0)}\n"
            reponse_summary += f"Points forts: {', '.join(pf) or 'Aucun'}\n"
            reponse_summary += f"Points faibles: {', '.join(pfa) or 'Aucun'}"

            resume_json = {
                "question": all_questions[i],
                "audio": audio_summary,
                "face": face_summary,
                "reponse": reponse_summary,
                "points_forts": pf,
                "points_faibles": pfa,
                "emotion": audio_result.get('emotion', 'inconnue'),
                "confidence": round(audio_result.get('confidence', 0) * 100),
                "explication": xai_result,
                "explication_face": xai_result_face
            }

            analyses.append(resume_json)

        # 🎓 Génération du PDF final
        full_transcription = "\n\n".join([a['reponse'] for a in analyses])
        pdf_file = export_to_pdf({
            "reponse": full_transcription,
            "points_forts": flatten_list([a['points_forts'] for a in analyses]),
            "points_faibles": flatten_list([a['points_faibles'] for a in analyses]),
            "emotion": "mixte",
            "confidence": sum(a['confidence'] for a in analyses) // N_QUESTIONS
        })
        
        explication_html = analyses[-1].get("explication", "<p>Aucune explication disponible.</p>")
        explication_face_html = analyses[-1].get("explication_face", "<p>Aucune explication visage.</p>"),
        explain_global_text_model()
        explain_global_face_model(all_frames)
        # Lister les heatmaps générées
        from pathlib import Path

        # 1. On construit un chemin absolu (plus sûr pour Gradio)
        base_dir = Path.cwd()  # Ou Path(__file__).parent.resolve() si tu es sûr que __file__ est défini

        emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        # 2. Liste les images et affiche dans la console celles trouvées ou non
        global_face_images = []
        for label in emotion_labels:
            img_path = base_dir / "outputs" / f"lime_face_global_{label}.png"
            if img_path.exists():
                print(f"✅ Image trouvée : {img_path}")
                global_face_images.append(str(img_path))
            else:
                print(f"❌ Image manquante : {img_path}")

        
        from pathlib import Path

        audio_xai_path = Path(f"outputs/audio_xai_q{N_QUESTIONS}.png")
        if not audio_xai_path.exists():
            audio_xai_path = None


        yield all_frames[-1], "Entretien terminé", "Analyse complète ci-dessous", analyses[-1], full_transcription, pdf_file, explication_html, explication_face_html, "summary_plot.png", global_face_images

    except Exception as e:
        print(f"Erreur globale: {e}")
        yield None, "Erreur", str(e), None, None, None, None, None, None, None





def explain_audio_local(audio_np, model, processor, segment_duration=1.0, sampling_rate=16000):
    import torch
    segment_samples = int(segment_duration * sampling_rate)
    total_samples = len(audio_np)
    num_segments = total_samples // segment_samples

    inputs = processor(audio_np, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        orig_logits = model(**inputs).logits
        orig_conf = float(orig_logits.softmax(dim=1)[0][1])

    impacts = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples

        muted_audio = audio_np.copy()
        muted_audio[start:end] = 0.0

        inputs = processor(muted_audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            conf = float(logits.softmax(dim=1)[0][1])

        delta = orig_conf - conf
        impacts.append(delta)

    return impacts, orig_conf

def plot_audio_explanation(audio_np, impacts, output_path="audio_xai_plot.png", segment_duration=1.0, sampling_rate=16000):
    import matplotlib.pyplot as plt
    times = np.arange(len(audio_np)) / sampling_rate
    plt.figure(figsize=(12, 4))
    plt.plot(times, audio_np, alpha=0.6)

    segment_samples = int(segment_duration * sampling_rate)
    max_impact = max(abs(i) for i in impacts) + 1e-6

    for i, impact in enumerate(impacts):
        start = i * segment_samples / sampling_rate
        end = (i + 1) * segment_samples / sampling_rate
        alpha = abs(impact) / max_impact
        color = "red" if impact > 0 else "blue"
        plt.axvspan(start, end, color=color, alpha=alpha * 0.5)

    plt.title("XAI locale : impact des segments audio")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()







import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import base64



from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def explain_response_with_lime(text):
    # Liste de stopwords à exclure
    stopwords = [
    "à", "abord", "afin", "ah", "alors", "au", "aucun", "aucune", "aujourd", "auprès", "avec", "avoir", "bon", "car", "ce", "cela", "celui", "cependant", "chaque", "ci", 
    "comme", "comment", "dans", "de", "debout", "dedans", "dehors", "depuis", "derrière", "des", "donc", "dont", "du", "duquel", "dès", "elle", "elles", "en", "encore", 
    "entre", "envers", "est", "étaient", "étant", "être", "eu", "fait", "faire", "fais", "faisons", "faites", "il", "ils", "je", "la", "le", "les", "leur", "leurs", "là", 
    "lors", "lorsque", "ma", "maintenant", "mais", "me", "mes", "mine", "moins", "mon", "mot", "même", "ni", "nommé", "notamment", "nous", "notre", "nôtre", "nous-mêmes", 
    "où", "par", "parce", "parce que", "pas", "pendant", "peut", "peut-être", "pour", "pourquoi", "près", "proche", "quand", "que", "quel", "quelle", "quelque", "quels", 
    "quelles", "qui", "quoi", "que", "quiconque", "rien", "sans", "se", "seulement", "si", "s'il", "sous", "sur", "surtout", "ta", "tant", "telles", "tes", "toi", "toi-même", 
    "tous", "tout", "toute", "toutes", "tu", "un", "une", "unes", "votre", "votre", "vous", "vous-mêmes", "vraiment", "voilà", "vos", "votre", "yeux", "y", "ça", "étais", 
    "était", "étions", "été", "être", "sont", "seront", "ceci", "celles", "celle", "quelques", "cela", "ceux", "celles", "tandis", "très", "quand", "enfin", "encore", "a", "et"
]

    
    # Filtrer les stopwords du texte
    filtered_text = " ".join([word for word in text.split() if word not in stopwords])

    # Données d'entraînement factices
    X_train = [
    "Je suis très motivé et j’aime travailler en équipe.",
    "Je suis désorganisé et je n’aime pas les responsabilités.",
    "J’ai une grande expérience en développement logiciel.",
    "Je ne sais pas coder et je suis souvent en retard.",
    "Je maîtrise les principes SOLID et la revue de code.",
    "Je ne connais pas les bases de Git ni les tests unitaires.",
    "Je documente mes projets et fais des tests régulièrement.",
    "Je suis souvent dépassé par les tâches techniques complexes.",
]
    y_train = [1, 0, 1, 0, 1, 0, 1, 0]

    # Pipeline de traitement
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
    pipeline.fit(X_train, y_train)

    # Explication avec LIME
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=["Mauvais", "Bon"])
    exp = explainer.explain_instance(filtered_text, pipeline.predict_proba, num_features=6)

    # Sauvegarde du graphique en image
    import tempfile
    import matplotlib.pyplot as plt

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        fig.savefig(tmpfile.name)
        plt.close(fig)
        tmpfile_path = tmpfile.name

    # Lecture et conversion en base64
    with open(tmpfile_path, "rb") as f:
        image_data = f.read()

    os.remove(tmpfile_path)
    encoded = base64.b64encode(image_data).decode("utf-8")
    img_html = f'<img src="data:image/png;base64,{encoded}" />'

    return img_html


"""
def explain_response_with_lime(text):
    # 🔧 Données d'entraînement factices
    X_train = [
    "Je suis très motivé et j’aime travailler en équipe.",
    "Je suis désorganisé et je n’aime pas les responsabilités.",
    "J’ai une grande expérience en développement logiciel.",
    "Je ne sais pas coder et je suis souvent en retard.",
    "Je maîtrise les principes SOLID et la revue de code.",
    "Je ne connais pas les bases de Git ni les tests unitaires.",
    "Je documente mes projets et fais des tests régulièrement.",
    "Je suis souvent dépassé par les tâches techniques complexes.",
    ]
    y_train = [1, 0, 1, 0, 1, 0, 1, 0]

    # 🔁 Entraînement d'un modèle simple
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
    pipeline.fit(X_train, y_train)


    # 🧠 Explication avec LIME
    explainer = LimeTextExplainer(class_names=["Mauvais", "Bon"])
    exp = explainer.explain_instance(text, pipeline.predict_proba, num_features=6)

    # 🖼️ Sauvegarde dans un fichier temporaire de manière sécurisée
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        fig.savefig(tmpfile.name)
        plt.close(fig)  # ferme proprement la figure pour libérer le fichier
        tmpfile_path = tmpfile.name

    # 📤 Lecture et conversion en base64 après fermeture complète
    with open(tmpfile_path, "rb") as f:
        image_data = f.read()

    os.remove(tmpfile_path)  # maintenant on peut supprimer sans verrouillage
    encoded = base64.b64encode(image_data).decode("utf-8")
    img_html = f'<img src="data:image/png;base64,{encoded}" />'
    
    return img_html
"""


import shap

def explain_global_text_model():
    X_train = [
        "Je suis très motivé et j’aime travailler en équipe.",
        "Je suis désorganisé et je n’aime pas les responsabilités.",
        "J’ai une grande expérience en développement logiciel.",
        "Je ne sais pas coder et je suis souvent en retard.",
    ]
    y_train = [1, 0, 1, 0]

    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
    pipeline.fit(X_train, y_train)

    explainer = shap.Explainer(pipeline.named_steps['logisticregression'],
                                pipeline.named_steps['tfidfvectorizer'].transform(X_train))
    shap_values = explainer(pipeline.named_steps['tfidfvectorizer'].transform(X_train))

    import matplotlib.pyplot as plt
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig("summary_plot.png")
    plt.close()


from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
import cv2
import tempfile
import base64
import os

def crop_face_for_lime(img_bgr):
    try:
        # DeepFace attend une image RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = DeepFace.extract_faces(img_rgb, detector_backend="opencv", enforce_detection=False)
        if faces and 'face' in faces[0]:
            face_array = (faces[0]['face'] * 255).astype(np.uint8)
            return face_array
        else:
            return img_rgb  # fallback : image complète
    except Exception as e:
        print(f"[LIME] Erreur de détection du visage : {e}")
        return img_bgr

def explain_face_with_lime(image_bgr):
    explainer = lime_image.LimeImageExplainer()

    face_img = crop_face_for_lime(image_bgr)  # format RGB

    def predict_fn(imgs):
        results = []
        for img in imgs:
            try:
                result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)[0]
                probs = result["emotion"]
                ordered = [probs[k] for k in sorted(probs.keys())]
                results.append(ordered)
            except Exception as e:
                print(f"[LIME] DeepFace error: {e}")
                results.append([0]*7)
        return np.array(results)

    try:
        explanation = explainer.explain_instance(
            face_img,
            classifier_fn=predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=1000
        )

        label = explanation.top_labels[0]
        img, mask = explanation.get_image_and_mask(
            label,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )

        fig, ax = plt.subplots()
        ax.imshow(mark_boundaries(img, mask))
        ax.axis("off")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name, bbox_inches='tight')
            temp_file_path = tmp.name
        plt.close(fig)

        with open(temp_file_path, "rb") as f:
            img_data = f.read()
        os.remove(temp_file_path)
        encoded = base64.b64encode(img_data).decode("utf-8")
        return f'<img src="data:image/png;base64,{encoded}" />'

    except Exception as e:
        print(f"[LIME] Erreur : {e}")
        return "<p>Explication LIME indisponible.</p>"



def explain_global_face_model(face_images):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    os.makedirs("outputs", exist_ok=True)

    explainer = lime_image.LimeImageExplainer()
    emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    importance_sums = {label: np.zeros((224, 224)) for label in emotion_labels}
    count_per_label = {label: 0 for label in emotion_labels}

    for img in face_images:
        try:
            img_resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))

            def predict_fn(imgs):
                results = []
                for i in imgs:
                    try:
                        result = DeepFace.analyze(i, actions=['emotion'], enforce_detection=False)[0]
                        probs = result["emotion"]
                        ordered = [probs[k] for k in sorted(probs.keys())]
                        results.append(ordered)
                    except:
                        results.append([0]*7)
                return np.array(results)

            explanation = explainer.explain_instance(
                img_resized,
                classifier_fn=predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000
            )

            label_idx = explanation.top_labels[0]
            label = emotion_labels[label_idx]
            _, mask = explanation.get_image_and_mask(label_idx, positive_only=True, num_features=10, hide_rest=False)

            importance_sums[label] += mask.astype(np.float32)
            count_per_label[label] += 1

        except Exception as e:
            print(f"[Global LIME Face] Erreur : {e}")

    for label in emotion_labels:
        if count_per_label[label] > 0:
            avg_mask = importance_sums[label] / count_per_label[label]
            plt.figure()
            plt.title(f"Importance moyenne : {label}")
            plt.imshow(avg_mask, cmap='hot')
            plt.axis("off")
            output_path = os.path.join("outputs", f"lime_face_global_{label}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            print(f"✅ Sauvegardé : {output_path}")




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
                explication_html = gr.HTML(label="Explication LIME")
                explication_face_html = gr.HTML(label="Explication LIME (Visage)")
                global_text_plot = gr.Image(label="Explication Globale (Texte)", value="summary_plot.png")
                global_face_gallery = gr.Gallery(label="Explication Globale (Visage)")
            status = gr.Textbox(label="Statut")        

        gr.Button("Démarrer l'interview").click(
    run_interview,
    outputs=[webcam, question, status, results, transcription_box, pdf_output, explication_html, explication_face_html, global_text_plot, global_face_gallery]
)



demo.launch(share=False)





"""
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

"""