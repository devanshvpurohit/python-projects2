# suradas.py - Gemini 1.5 AI Assistant for the Visually Impaired

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
# Voice Command
# --------------------------
def listen_and_process():
    st.subheader("🎙️ Speak Now")
    audio = mic_recorder(start_prompt="Click to record", stop_prompt="Stop", just_once=True, key="mic")

    if audio:
        try:
            recognizer = sr.Recognizer()
            audio_data = sr.AudioData(audio["bytes"], sample_rate=audio["sample_rate"], sample_width=2)
            text = recognizer.recognize_google(audio_data)
            speak(f"You said: {text}")
            return text
        except Exception:
            speak("Sorry, I couldn't understand the audio.")
            return None

# --------------------------
# Video Frame Grabber
# --------------------------
class VideoCapture(VideoTransformerBase):
    def __init__(self):
        self.last_frame = None

    def transform(self, frame):
        self.last_frame = frame.to_image()
        return frame

# --------------------------
# Image to Bytes
# --------------------------
def image_to_bytes(img):
    buf = BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()

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
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Suradas", layout="centered")
st.title("🎧 Suradas — Gemini AI for the Visually Impaired")

if "history" not in st.session_state:
    st.session_state.history = []

ctx = webrtc_streamer(key="camera", video_transformer_factory=VideoCapture)

command = listen_and_process()

if command:
    st.session_state.history.append(command)
    command = command.lower()

    # Normal Search
    if "search for" in command:
        query = command.split("search for")[-1].strip()
        response = model.generate_content(f"Answer this: {query}")
        speak(response.text)
        st.write(response.text)

    # Object Detection
    elif "object" in command or "what is in front" in command:
        if ctx.video_transformer and ctx.video_transformer.last_frame:
            frame = ctx.video_transformer.last_frame
            result = gemini_vision_task(frame, "Describe all visible objects.")
            st.write(result)

    # Human Detection
    elif "person" in command or "human" in command:
        if ctx.video_transformer and ctx.video_transformer.last_frame:
            frame = ctx.video_transformer.last_frame
            result = gemini_vision_task(frame, "Is there any human or person in the image?")
            st.write(result)

    # Currency Detection
    elif "currency" in command:
        if ctx.video_transformer and ctx.video_transformer.last_frame:
            frame = ctx.video_transformer.last_frame
            result = gemini_vision_task(frame, "What currency and denomination is this?")
            st.write(result)

    # Translation
    elif "translate to" in command:
        target_lang = command.split("translate to")[-1].strip()
        speak(f"Say the text you want to translate to {target_lang}")
        text_to_translate = listen_and_process()
        if text_to_translate:
            translated = gemini_translate(text_to_translate, target_lang)
            st.write(translated)

    # Geolocation
    elif "where am i" in command or "location" in command:
        location = describe_location()
        st.write(location)

    else:
        speak("Command not recognized. Try saying search, object, person, currency, translate, or location")

# --------------------------
# Command History
# --------------------------
st.divider()
st.subheader("🧠 Command History")
for cmd in st.session_state.history:
    st.write("→", cmd)
