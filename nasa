import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(page_title="Streamlit Call App", layout="centered")

st.title("📞 Streamlit WebRTC Video Call")

st.markdown("""
Welcome! This is a peer-to-peer browser-based video call app built with WebRTC and Streamlit.

- ✅ Works on Streamlit Cloud
- 🔐 Peer-to-peer (via WebRTC)
- 🌐 Uses public STUN servers
- 🧪 Run in 2 tabs or share your deployed link
""")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        return frame  # No processing; just return raw frame

webrtc_streamer(
    key="example",
    video_processor_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": True},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
