import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the pre-trained model from TensorFlow Hub
model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
model = tf.keras.Sequential([hub.KerasLayer(model_url)])

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to match the model's input size
    image = np.asarray(image) / 255.0  # Normalize pixel values
    return image

# Function to predict plant species
def predict_plant_species(image):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    return predictions

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for processing the uploaded image
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = cv2.imread(image_path)
        predictions = predict_plant_species(image)
        os.remove(image_path)  # Remove the uploaded image after processing
        return str(predictions)
    else:
        return "Error: Invalid file format"

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True)
