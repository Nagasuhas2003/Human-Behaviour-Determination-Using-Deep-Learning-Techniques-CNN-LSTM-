import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

# Configuration
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
SELECTED_ACTIONS = [
    'Basketball', 'Boxing', 'CliffDiving', 
    'CricketShot', 'GolfSwing', 'JumpRope', 
    'Shotput', 'Surfing', 'TennisSwing', 'Volleyball'
]

# Load the trained model
model = tf.keras.models.load_model('action_recognition_model.h5')

def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < SEQUENCE_LENGTH:
        return []
    
    skip_window = max(total_frames // SEQUENCE_LENGTH, 1)
    
    for i in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, i * skip_window)
        success, frame = video_reader.read()
        if not success:
            break
        
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        frames_list.append(frame / 255.0)
    
    video_reader.release()
    return frames_list if len(frames_list) == SEQUENCE_LENGTH else []

def predict_action(video_path):
    frames = frames_extraction(video_path)
    
    if len(frames) != SEQUENCE_LENGTH:
        st.error(f"Error: Extracted {len(frames)} frames (need {SEQUENCE_LENGTH})")
        return None
    
    input_data = np.expand_dims(frames, axis=0)
    predictions = model.predict(input_data)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]
    
    return [(SELECTED_ACTIONS[i], float(predictions[i])) for i in top_indices]

# Streamlit UI
st.set_page_config(page_title="Human Behaviour Determination", page_icon="🎥")
st.title("Human Behaviour Determination")

# Dynamic Background
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://example.com/your-background-image.jpg'); /* Replace with your image URL */
        background-size: cover;
        background-position: center;
        color: white;
    }
    .top-predictions {
        font-size: 24px; /* Increase font size for top predictions */
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Input Section
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save the uploaded video
    video_path = os.path.join("uploads", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display video
    st.video(video_path)

    # Recognize Action Button
    if st.button("Recognize Action"):
        with st.spinner("Processing..."):
            results = predict_action(video_path)
            if results:
                st.success("Top Predictions:", icon="✅")
                for action, confidence in results:
                    st.markdown(f"<div class='top-predictions'>- {action}: {confidence:.2%}</div>", unsafe_allow_html=True)
                # Hide input elements after processing
                st.session_state.show_input = False
            else:
                st.error("Prediction failed.")

# Control visibility of input elements
if 'show_input' not in st.session_state:
    st.session_state.show_input = True

if st.session_state.show_input:
    st.write("Upload a video and click 'Recognize Action' to see results.")
