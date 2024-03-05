import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from tkinter import Tk, Button, filedialog, Label
import os

# Load the pre-trained model from TensorFlow Hub
model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
model = tf.keras.Sequential([hub.KerasLayer(model_url)])

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize to match the model's input size
    image = np.asarray(image) / 255.0  # Normalize pixel values
    return image

# Function to predict plant species
def predict_plant_species(image_path):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    return predictions

# Function to process the uploaded or captured image
def process_image(image_path):
    predictions = predict_plant_species(image_path)
    print(predictions)

# Function to capture image from camera and store in "Training" folder
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        cap.release()
        return

    cap.release()
    cv2.destroyAllWindows()

    # Create the "Training" folder if it doesn't exist
    if not os.path.exists("Training"):
        os.makedirs("Training")

    # Save the captured image in the "Training" folder
    image_path = os.path.join("Training", "captured_image.jpg")
    cv2.imwrite(image_path, frame)

    process_image(image_path)

# Function to upload image using Tkinter
def upload_image():
    root = Tk()
    root.filename = filedialog.askopenfilename(title="Select an image file",
                                               filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))
    root.withdraw()
    if root.filename:
        process_image(root.filename)

# Create a simple GUI
root = Tk()
root.title("Plant Detection")
capture_button = Button(root, text="Capture Image", command=capture_image)
capture_button.pack()
upload_button = Button(root, text="Upload Image", command=upload_image)
upload_button.pack()
root.mainloop()
