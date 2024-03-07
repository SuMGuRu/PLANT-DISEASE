import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import requests
import os

# Download VGG16 model weights if not already downloaded
model_path = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
    r = requests.get(url, allow_redirects=True)
    open(model_path, 'wb').write(r.content)

# Load the VGG16 model with pre-trained weights
model = VGG16(weights=model_path)

# Function to predict the image class using the VGG16 model
def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]

# Streamlit app code
st.title("VGG16 Image Classification")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    labels = predict_image(uploaded_file, model)
    for label in labels:
        st.write(f"{label[1]}: {label[2]*100:.2f}%")
