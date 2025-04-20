import streamlit as st
import torch
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import google.generativeai as genai
from PIL import Image

# --- HARD-CODED API KEY (use only for private testing) ---
GOOGLE_API_KEY = "AIzaSyA5nrHFYD3u6nXnj70POa7bc8qr0vB9qKA"
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro-vision")

# Load YOLOv5 model from GitHub
@st.cache_resource
def load_model():
    return torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

model_yolo = load_model()
model_yolo.eval()

# Streamlit page setup
st.set_page_config(page_title="YOLO + Gemini Vision", layout="wide")
st.title("🔍 Real-time Object Detection with Gemini Pro Vision")
st.sidebar.title("📊 Gemini Object Insights")
sidebar_placeholder = st.sidebar.empty()

# Gemini analysis function
def analyze_object(img_pil, class_name):
    prompt = f"Analyze this object. What is it, and what are its potential uses or characteristics? This is a '{class_name}'."
    try:
        response = gemini_model.generate_content([prompt, img_pil])
        return response.text
    except Exception as e:
        return f"Error: {e}"

# YOLOv5 + Gemini processor
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_gemini_outputs = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = model_yolo(img_rgb)
        self.latest_gemini_outputs.clear()

        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            class_name = model_yolo.names[int(cls)]

            crop_img = img[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue

            pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            gemini_response = analyze_object(pil_img, class_name)
            self.latest_gemini_outputs.append((class_name, gemini_response))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start stream
ctx = webrtc_streamer(
    key="yolo-gemini-stream",
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Show Gemini results
if ctx.video_processor:
    with st.sidebar:
        st.markdown("### Latest Gemini Results")
        for cls_name, analysis in ctx.video_processor.latest_gemini_outputs:
            with st.expander(f"🔎 {cls_name}"):
                st.write(analysis)
