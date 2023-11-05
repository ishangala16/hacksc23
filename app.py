import streamlit as st
import pandas as pd 
import numpy as np
import pickle
from PIL import Image

#Beautification
st.set_page_config(page_title="AI Wound Analysis", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")


# Load model
# model = pickle.load(open('model.pkl', 'rb')) 

# Create title
st.title('Wound Imaging Analysis Web App')

# Upload image
image = st.file_uploader('Upload wound image', type=['jpg','png'])

if image is not None:

    # Load image
    img = Image.open(image)

    # Preprocess image
    
    # Make prediction
    # prediction = model.predict(img)
    
    # Show image and prediction
    st.image(img)
    st.success(f'Prediction: {prediction}')
    
else:
    st.info('Upload an image')