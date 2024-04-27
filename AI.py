import tkinter as tk
from tkinter import Tk, Button, Label, Entry, filedialog
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import requests
from bs4 import BeautifulSoup
import sqlite3
import cv2
import numpy as np
from keras.applications import resnet50
from keras.preprocessing import image
from keras.applications.resnet50 import decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import imghdr

# Create a Tkinter window
window = tk.Tk()

# Create a file upload button
upload_button = tk.Button(window, text="Upload Image", command=lambda: upload_file())
upload_button.pack()

# Create a text box for the plant name
plant_name_label = tk.Label(window, text="Plant Name:")
plant_name_label.pack()
plant_name_entry = tk.Entry(window)
plant_name_entry.pack()

# Create a SQLite3 connection
conn = sqlite3.connect('plants.db')
conn.execute('''CREATE TABLE IF NOT EXISTS plants
             (name TEXT, scientific_name TEXT, habitat TEXT, fruits TEXT, common_names TEXT)''')

# Define a function to handle file uploads
def upload_file():
    plant_name = plant_name_entry.get()

    # Open a file dialog and get the file path
    file_path = filedialog.askopenfilename()

    # Open the file using imghdr to determine the image format and size
    with open(file_path, 'rb') as f:
        image_format = imghdr.what(f)
        image_size = len(f.read())
        f.seek(0)

    # Load the image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)

    # Preprocess the image for input to the CNN model
    if image_format == 'jpeg':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image_format == 'png':
        pass
    else:
        raise ValueError(f"Unsupported image format: {image_format}")
    image = cv2.resize(image, (256, 256))
    image = image.astype("float32") / 255
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Use the pre-trained ResNet50 model to classify the image
    model = ResNet50(weights='imagenet')
    predictions = model.predict(image)
    top_predictions = decode_predictions(predictions, top=3)[0]

    # Display the top 3 predictions to the user
    print("Top 3 predictions:")
    for i, prediction in enumerate(top_predictions):
        print(f"{i+1}. {prediction[1]}: {prediction[2]*100:.2f}%")

        # Scrape plant information from relevant websites
        plant_type = prediction[1]
        url = f"https://www.google.com/search?q={plant_type}+information"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')


        # Extract plant information from the parsed HTML
        scientific_name = soup.find("span", {"id": "scientific-name"}).text
        habitat = soup.find("p", {"class": "habitat"}).text
        fruits = soup.find("span", {"id": "fruits"}).text
        common_names = soup.find_all("span", {"class": "common-name"})

        # Print the extracted plant information
        print(f"\nName: {plant_name}")
        print(f"Scientific Name: {scientific_name}")
        print(f"Habitat: {habitat}")
        print(f"Fruits: {fruits}")
        print("Common Names:")
        for common_name in common_names:
            print(common_name.text)

        # Store the scraped plant information in a SQLite3 database
        conn = sqlite3.connect('plants.db')

def get_plant_info(plant_name):
    url = f"https://www.google.com/search?q={plant_name}+information"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    scientific_name = soup.find("span", {"id": "scientific-name"}).text
    habitat = soup.find("p", {"class": "habitat"}).text
    fruits = soup.find("span", {"id": "fruits"}).text
    common_names = soup.find_all("span", {"class": "common-name"})

    plant_info = {
        "name": plant_name,
        "scientific_name": scientific_name,
        "habitat": habitat,
        "fruits": fruits,
        "common_names": [common_name.text for common_name in common_names]
    }

    return plant_info

def classify_image():
    # Load the image
    image_path = image_entry.get()
    image = cv2.imread(image_path)

    # Classify the image using a pre-trained CNN model
    model = cv2.dnn.readNetFromCaffe("resnet50.prototxt", "resnet50.caffemodel")
    blob = cv2.dnn.blobFromImage(image, 1, (256, 256), (104, 117, 123))
    model.setInput(blob)
    predictions = model.forward()

    # Get the top prediction
    top_prediction = predictions[0].argmax()
    top_prediction_name = "prediction_names.txt"
    with open(top_prediction_name, "r") as f:
        prediction_names = f.readlines()

    plant_name = prediction_names[top_prediction].strip()

    # Get the plant information
    plant_info = get_plant_info(plant_name)

    # Print the plant information
    print(f"\nName: {plant_info['name']}")
    print(f"Scientific Name: {plant_info['scientific_name']}")
    print(f"Habitat: {plant_info['habitat']}")
    print(f"Fruits: {plant_info['fruits']}")
    print("Common Names:")
    for common_name in plant_info['common_names']:
        print(common_name)

root = Tk()

# Create a label and entry for the image path
image_label = Label(root, text="Image Path:")
image_label.pack()
image_entry = Entry(root)
image_entry.pack()

# Create a button to classify the image
classify_button = Button(root, text="Classify Image", command=classify_image)
classify_button.pack()

root.mainloop()