# suradas.py — AI Assistant for the Visually Impaired using Gemini 1.5

import streamlit as st
import pyttsx3
import google.generativeai as genai
import requests
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from io import BytesIO

# --------------------------
# Gemini 1.5 Configuration
# --------------------------
genai.configure(api_key="AIzaSyA0INYcsqw8dkI9KbEB7jt4l7hafoLDNW4")
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
# Voice or Text Command Input
# --------------------------
def get_command():
    st.subheader("🎙️ Speak or type your command")
    audio = mic_recorder(start_prompt="Click to record", stop_prompt="Stop", just_once=True, key="mic")
    text_input = st.text_input("Or type your command below")

    if audio:
        try:
            recognizer = sr.Recognizer()
            audio_data = sr.AudioData(audio["bytes"], sample_rate=audio["sample_rate"], sample_width=2)
            text = recognizer.recognize_google(audio_data)
            speak(f"You said: {text}")
            return text
        except:
            speak("Sorry, I couldn't understand the audio.")
            return None
    elif text_input:
        return text_input
    return None

# --------------------------
# Camera Stream Capture
# --------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_frame = None

    def transform(self, frame):
        self.last_frame = frame.to_image()
        return frame

def image_to_bytes(img):
    buf = BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()

# --------------------------
# Gemini Vision Task
# --------------------------
def gemini_vision(image, prompt):
    try:
        image_bytes = image_to_bytes(image)
        response = vision_model.generate_content([prompt, image_bytes])
        speak(response.text)
        return response.text
    except Exception as e:
        speak("Vision processing error.")
        return f"Error: {e}"

# --------------------------
# Normal Search
# --------------------------
def gemini_search(query):
    try:
        response = model.generate_content(query)
        speak(response.text)
        return response.text
    except Exception as e:
        speak("Search failed.")
        return f"Error: {e}"

# --------------------------
# Translation
# --------------------------
def gemini_translate(text, lang):
    prompt = f"Translate this to {lang}: {text}"
    return gemini_search(prompt)

# --------------------------
# Geolocation
# --------------------------
def describe_location():
    try:
        data = requests.get("https://ipapi.co/json").json()
        location_info = f"IP: {data['ip']}, City: {data['city']}, Region: {data['region']}, Country: {data['country_name']}"
        prompt = f"Describe this location for a visually impaired person: {location_info}"
        return gemini_search(prompt)
    except Exception as e:
        speak("Location retrieval failed.")
        return f"Error: {e}"

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Suradas", layout="centered")
st.title("🎧 Suradas — Gemini 1.5 AI for the Visually Impaired")

if "history" not in st.session_state:
    st.session_state.history = []

ctx = webrtc_streamer(key="live-camera", video_transformer_factory=VideoProcessor)

command = get_command()

if command:
    st.session_state.history.append(command)

    # Object Detection
    if "object" in command.lower():
        speak("Describing visible objects...")
        if ctx.video_transformer and ctx.video_transformer.last_frame:
            result = gemini_vision(ctx.video_transformer.last_frame, "Describe all visible objects.")
            st.write(result)

    # Human Detection
    elif "human" in command.lower() or "person" in command.lower():
        speak("Checking for human presence...")
        if ctx.video_transformer and ctx.video_transformer.last_frame:
            result = gemini_vision(ctx.video_transformer.last_frame, "Is there any person in this image?")
            st.write(result)

    # Translation
    elif "translate" in command.lower() and "to" in command.lower():
        parts = command.lower().split("to")
        target_lang = parts[-1].strip()
        speak(f"Say the text you want to translate to {target_lang}.")
        source_text = get_command()
        if source_text:
            result = gemini_translate(source_text, target_lang)
            st.write(result)

    # Location
    elif "location" in command.lower() or "where am i" in command.lower():
        speak("Describing your current location.")
        result = describe_location()
        st.write(result)

    # Search
    elif "search" in command.lower():
        speak("Say what you'd like to search.")
        query = get_command()
        if query:
            result = gemini_search(query)
            st.write(result)

    else:
        speak("Command not recognized. Try object, human, translate, location, or search.")

# --------------------------
# Command History
# --------------------------
st.divider()
st.subheader("🧠 Past Commands")
for cmd in st.session_state.history:
    st.write("→", cmd)
