# suradas.py — AI Mockup for the Blind using Gemini 1.5

import streamlit as st
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
import requests
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --------------------------
# Gemini 1.5 Configuration
# --------------------------
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
vision_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-vision")

# --------------------------
# Text to Speech
# --------------------------
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# --------------------------
# Voice Command
# --------------------------
def listen_and_process():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening for your command")
        audio = r.listen(source, timeout=5)
    try:
        text = r.recognize_google(audio)
        speak(f"You said: {text}")
        return text
    except:
        speak("Sorry, I didn't catch that.")
        return None

# --------------------------
# Camera Frame Grabber
# --------------------------
class VideoCapture(VideoTransformerBase):
    def transform(self, frame):
        self.last_frame = frame.to_image()
        return frame

# --------------------------
# Gemini Vision Task
# --------------------------
def gemini_vision_task(image, prompt):
    try:
        image_bytes = image_to_bytes(image)
        response = vision_model.generate_content([prompt, image_bytes])
        speak(response.text)
        return response.text
    except Exception as e:
        speak("There was an error processing the image")
        return f"Error: {e}"

def image_to_bytes(img):
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()

# --------------------------
# Translation
# --------------------------
def gemini_translate(text, target_lang):
    prompt = f"Translate this to {target_lang}: {text}"
    try:
        response = model.generate_content(prompt)
        speak(response.text)
        return response.text
    except Exception as e:
        speak("Translation failed.")
        return f"Error: {e}"

# --------------------------
# Geolocation
# --------------------------
def describe_location():
    try:
        data = requests.get("https://ipapi.co/json").json()
        location_info = f"IP: {data['ip']}, City: {data['city']}, Region: {data['region']}, Country: {data['country_name']}"
        prompt = f"Give a friendly description of this location: {location_info}"
        response = model.generate_content(prompt)
        speak(response.text)
        return response.text
    except Exception as e:
        speak("Could not retrieve your location.")
        return f"Error: {e}"

# --------------------------
# UI + Session State
# --------------------------
st.set_page_config(page_title="Suradas", layout="centered", initial_sidebar_state="collapsed")
st.title("🎧 Suradas — Gemini 1.5-Powered AI for the Visually Impaired")

if "history" not in st.session_state:
    st.session_state.history = []

command = listen_and_process()

if command:
    st.session_state.history.append(command)

    # Object Detection
    if "object" in command.lower():
        speak("Object Detection Mode")
        ctx = webrtc_streamer(key="object-camera", video_transformer_factory=VideoCapture)
        if ctx.video_transformer:
            speak("Show the object and say 'capture' to analyze")
            if listen_and_process() == "capture":
                frame = ctx.video_transformer.last_frame
                result = gemini_vision_task(frame, "Describe all visible objects.")

    # Currency Detection
    elif "currency" in command.lower():
        speak("Currency Detection Mode")
        ctx = webrtc_streamer(key="currency-camera", video_transformer_factory=VideoCapture)
        if ctx.video_transformer:
            speak("Show the currency and say 'detect'")
            if listen_and_process() == "detect":
                frame = ctx.video_transformer.last_frame
                result = gemini_vision_task(frame, "What currency and denomination is this?")

    # Translation
    elif "translate" in command.lower():
        if "to" in command.lower():
            parts = command.lower().split("to")
            target_lang = parts[-1].strip()
            speak(f"Say the text you want to translate to {target_lang}")
            text_to_translate = listen_and_process()
            if text_to_translate:
                translated = gemini_translate(text_to_translate, target_lang)
        else:
            speak("Please say: Translate to [language]")

    # Location
    elif "where am i" in command.lower() or "location" in command.lower():
        speak("Getting your location")
        describe_location()

    else:
        speak("Command not recognized. Try saying object, currency, translate, or where am I")

# --------------------------
# Command History
# --------------------------
st.divider()
st.subheader("🧠 Past Commands (Session)")
for cmd in st.session_state.history:
    st.write("→", cmd)
