# suradas.py

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
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening...")
        audio = r.listen(source, timeout=5)
    try:
        text = r.recognize_google(audio)
        st.success(f"You said: {text}")
        speak(f"You said: {text}")
        return text
    except:
        st.error("Sorry, I didn't catch that.")
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
        return f"Error: {e}"

# --------------------------
# UI + Session State
# --------------------------
st.set_page_config(page_title="Suradas", layout="centered")
st.title("🎧 Suradas — Gemini 1.5-Powered AI Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("#### Say a command like:")
st.code("""
• Detect object
• Detect currency
• Translate to Hindi
• Where am I
""")

command = None

if st.button("🎙️ Start Listening"):
    try:
        command = listen_and_process()
    except:
        st.warning("🎤 Microphone not available. Please type your command below.")

if not command:
    command = st.text_input("⌨️ Type your command here:")

if command:
    st.session_state.history.append(command)

    # Object Detection
    if "object" in command.lower():
        st.subheader("📸 Object Detection via Camera")
        ctx = webrtc_streamer(key="object-camera", video_transformer_factory=VideoCapture)
        if ctx.video_transformer:
            st.warning("Click the button after showing the object")
            if st.button("🖼️ Capture Frame"):
                frame = ctx.video_transformer.last_frame
                result = gemini_vision_task(frame, "Describe all visible objects.")
                st.info(result)

    # Currency Detection
    elif "currency" in command.lower():
        st.subheader("💵 Currency Detection via Camera")
        ctx = webrtc_streamer(key="currency-camera", video_transformer_factory=VideoCapture)
        if ctx.video_transformer:
            st.warning("Show currency clearly before capturing")
            if st.button("💸 Detect Currency"):
                frame = ctx.video_transformer.last_frame
                result = gemini_vision_task(frame, "What currency and denomination is this?")
                st.info(result)

    # Translation
    elif "translate" in command.lower():
        st.subheader("🌐 Translation")
        if "to" in command.lower():
            parts = command.lower().split("to")
            target_lang = parts[-1].strip()
            text = st.text_area("Enter text to translate")
            if st.button("Translate"):
                translated = gemini_translate(text, target_lang)
                st.success(translated)
        else:
            st.warning("Please say: Translate to [language]")

    # Location
    elif "where am i" in command.lower() or "location" in command.lower():
        st.subheader("📍 Your Location")
        location = describe_location()
        st.info(location)

    else:
        st.warning("Command not recognized.")

# --------------------------
# Command History
# --------------------------
st.divider()
st.subheader("🧠 Past Commands (Session)")
for cmd in st.session_state.history:
    st.write("→", cmd)
