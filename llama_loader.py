import gradio as gr
import pyttsx3
import time
from threading import Thread
import random

# 📸 Avatar image path (remplace avec ton propre fichier si tu veux)
AVATAR_PATH = "avatar.png"

# 🎙️ Init synthèse vocale (local, pas besoin d'API)
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Vitesse de la voix

# 🔄 Liste d'exemples de questions
QUESTIONS = [
    "Peux-tu me parler d'un projet que tu as dirigé récemment ?",
    "Comment gères-tu les délais serrés dans un environnement agile ?",
    "Quels sont tes outils préférés pour le développement backend ?",
    "Comment sécuriser une API REST ?",
    "As-tu déjà résolu un conflit au travail ? Comment ?"
]

# 🔊 Fonction de lecture vocale
def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    t = Thread(target=run)
    t.start()

# 🚀 Fonction principale de génération d'une question
def poser_question():
    question = random.choice(QUESTIONS)
    speak(question)
    return AVATAR_PATH, question

# 🎨 Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Avatar IA - Recruteur interactif")
    
    with gr.Row():
        image_output = gr.Image(label="Recruteur", type="filepath", width=300, height=300)
        texte_output = gr.Textbox(label="Question posée", lines=2)

    bouton = gr.Button("🧠 Poser une question")

    bouton.click(fn=poser_question, outputs=[image_output, texte_output])

# 💥 Lancement
demo.launch()
