from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load the model
model = load_model('wild_animal_detection_model.h5')

# Define class labels
class_labels = ['elephant', 'gaur', 'leopard', 'lion', 'tiger']

def preprocess_image(img):
    img = img.resize((128, 128))  # Resize image to match model input size
    img = np.array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        img = preprocess_image(img)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_class_label = class_labels[predicted_class[0]]
        return jsonify({'prediction': predicted_class_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
