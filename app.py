import streamlit as st
import pandas as pd 
import numpy as np
import pickle
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
from torch import nn
import os

#Beautification
st.set_page_config(page_title="AI Wound Analysis", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.convnext_base(pretrained=True)
model.classifier = nn.Sequential(
    torchvision.models.convnext.LayerNorm2d((1024,), eps=1e-06, elementwise_affine=True),
    nn.Flatten(start_dim=1, end_dim=-1), 
    nn.Linear(in_features=1024, out_features=11, bias=True)
)
model.load_state_dict(torch.load('model_weights_epoch79.pth'))
model.eval()
model.to(device)

# Create title
st.title('Wound Imaging Analysis Web App')

# Upload image
image = st.file_uploader('Upload wound image', type=['jpg','png'])

def predict(model, input):
    with torch.no_grad():
        input = input.to(device)
        output = model(input)
        probabilities = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities)
        pred_label = {v: k for k, v in label_mapping.items()}[pred_class.item()]
    return pred_label

if image is not None:

    # Load image
    img = Image.open(image)
    image = transform(image).unsqueeze(0) 

    # Preprocess image
    
    # Make prediction
    pred = predict(model, image)
    
    # Show image and prediction
    st.image(img)
    st.success(f'Prediction: {pred}')
    
else:
    st.info('Upload an image')
