import streamlit as st
import google.generativeai as genai

# Configure the GenAI API
API_KEY = "AIzaSyA0INYcsqw8dkI9KbEB7jt4l7hafoLDNW4"
genai.configure(api_key=API_KEY)

# Load models
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
vision_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-vision")

# Page setup
st.set_page_config(page_title="Context-Aware Conversational AI", layout="wide")
st.title("🤖 AI for Context-Aware Conversational Agents")

# Explanation text
prompt = """
Explain in detail how AI is used to build context-aware conversational agents.
Include information about context-awareness, AI's role (NLP, ML), examples, and future directions.
Use a clear and engaging tone suitable for a beginner audience.
"""

# Generate content
if st.button("Generate Explanation with Gemini"):
    with st.spinner("Thinking..."):
        response = model.generate_content(prompt)
        st.markdown(response.text)

# Optional image input (future expansion with vision model)
st.markdown("---")
st.subheader("Want to explore visual AI too?")
image_input = st.file_uploader("Upload an image to analyze context with Vision model", type=["jpg", "jpeg", "png"])

if image_input:
    with st.spinner("Analyzing image context with Gemini Vision..."):
        image_bytes = image_input.read()
        vision_response = vision_model.generate_content(["Describe the context of this image in a few sentences.", image_bytes])
        st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
        st.markdown(vision_response.text)

