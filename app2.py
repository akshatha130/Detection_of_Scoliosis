import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# st.markdown(
#     """
#     <style>
#     .block-container {
#         max-width: 100%;
#         padding: 1rem;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Change Name & Logo
st.set_page_config(page_title="Scoliosis Prediction", page_icon="⚕️")

# Set environment variable for TensorFlow memory optimization
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"

# Load your trained model
MODEL_PATH = r"C:\Users\aksha\Downloads\scoliosis-detection-main\scoliosis-detection-main\Code\spine_classification_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Empower Your Health, Embrace Vitality🍀")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()  # Stop the app if the model fails to load

def preprocess_image(img):
    """
    Preprocess the image:
    - Resize to match model input shape (250, 950)
    - Convert to grayscale
    - Normalize pixel values to [0, 1]
    - Add batch and channel dimensions
    """
    img = img.resize((950, 250))  
    img = img.convert("L")  
    img = np.array(img) / 255.0 
    img = np.expand_dims(img, axis=0)  
    img = np.expand_dims(img, axis=-1)
    return img

# Streamlit UI
st.title("Scoliosis Detection App 🏥")
st.write("Upload an X-ray image to check if it is **Normal**, **Scoliosis**.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Open and display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img_processed = preprocess_image(img)

        # Make prediction
        prediction = model.predict(img_processed)
        predicted_class = np.argmax(prediction) 
        confidence = np.max(prediction) * 100 

        # Define classes
        classes = ["Normal", "Scoliosis", "Spondylosis"]

        # Display results
        st.write(f"### Prediction: **{classes[predicted_class]}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        # Provide feedback based on prediction
        if predicted_class == 1:
            st.error("⚠️ Scoliosis detected! Please consult a doctor.")
        elif predicted_class == 2:
            st.warning("⚠️ Spondylosis detected! Please consult a doctor.")
        else:
            st.success("✅ No abnormalities detected. Stay healthy!")
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
    st.write("AI can make Mistakes Please Make sure to Consult Doctor Before Taking Any Decision")