import streamlit as st
import google.generativeai as genai

# Configure the GenAI API
API_KEY = "AIzaSyA0INYcsqw8dkI9KbEB7jt4l7hafoLDNW4"
genai.configure(api_key=API_KEY)

# Load model
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")

# Page setup
st.set_page_config(page_title="Context-Aware Conversational AI", layout="wide")
st.title("🤖 AI for Context-Aware Conversational Agents")

# Text input for user-generated questions
user_input = st.text_area("Enter a question or topic related to context-aware conversational AI:", height=150)

# Generate content
if st.button("Generate Response"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            response = model.generate_content(user_input)
            st.markdown(response.text)
    else:
        st.warning("Please enter a question or topic to generate a response.")

# Footer
st.markdown("""
---
Made with ❤️ using [Streamlit](https://streamlit.io)
""")
