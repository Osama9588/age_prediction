from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError
import numpy as np
import streamlit as st

# Function to preprocess the image
def get_image_features(image):
    img = load_img(image, color_mode='grayscale')
    img = img.resize((128, 128), Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
    img = img_to_array(img)
    img = img.reshape(1, 128, 128, 1)
    img = img / 255.0
    return img

# Load the trained model
model = load_model('model.h5', custom_objects={'mae': MeanAbsoluteError()})

# Define gender mapping
gender_mapping = {
    1: 'Female',
    0: 'Male'
}

# Streamlit app
st.title("Age and Gender Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess and predict
    features = get_image_features(uploaded_file)
    pred = model.predict(features)
    
    # Extract predictions
    gender = gender_mapping[round(pred[0][0][0])]
    age = round(pred[1][0][0])

    # Display results
    st.write(f"Predicted Age: {age}")
    st.write(f"Predicted Gender: {gender}")
