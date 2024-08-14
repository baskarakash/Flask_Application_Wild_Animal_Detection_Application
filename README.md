# Wild Animal Detection Application

## Overview

The Wild Animal Detection Application is designed to classify images of wild animals using a Convolutional Neural Network (CNN). This project integrates a trained deep learning model with a web interface built using Flask. Users can upload images, and the application will predict the class of the animal in the image.

## Project Components

### 1. Machine Learning Model (Wild_Animal_Detection_ML.py)

**Purpose**: Train a CNN model to classify images of wild animals into categories: Elephant, Gaur, Leopard, Lion, and Tiger.

**Techniques and Tools**:
- **Convolutional Neural Networks (CNNs)**: Utilized for image classification due to their ability to capture spatial hierarchies in images.
- **Data Augmentation**: Not explicitly used in the provided code but can be beneficial for improving model performance by artificially increasing the size and variability of the training dataset.
- **Model Architecture**:
  - **Convolutional Layers**: Extract features from images. Configured with varying filter sizes and depths (32, 64, 128, 256) to capture different levels of abstraction.
  - **MaxPooling Layers**: Reduce spatial dimensions to prevent overfitting and decrease computational load.
  - **Dropout Layers**: Prevent overfitting by randomly dropping neurons during training.
  - **Dense Layers**: Fully connected layers for classification.
- **Optimizers**: Adam optimizer used for its adaptive learning rate capabilities.
- **Loss Function**: Categorical Crossentropy, suitable for multi-class classification problems.

**Data Preparation**:
- **Dataset Organization**: Images are stored in directories named after their respective classes. Training and validation data are separated into different directories.
- **Preprocessing**:
  - Resizing images to a fixed size of 128x128 pixels.
  - Normalizing pixel values to the range [0, 1].

**Output**:
- The model is saved as `wild_animal_detection_model.h5` for later use in the web application.

### 2. Web Application (app.py)

**Purpose**: Serve a web interface where users can upload images and get predictions from the trained model.

**Techniques and Tools**:
- **Flask**: A lightweight web framework for serving the web interface and handling HTTP requests.
- **File Handling**: Processes image uploads through Flask’s request handling system.
- **Image Preprocessing**:
  - Resize images to 128x128 pixels.
  - Normalize pixel values.
- **Model Inference**:
  - Load the trained model.
  - Predict the class of the uploaded image.
  - Return the prediction as a JSON response.

**Endpoints**:
- `/`: Renders the HTML page with the image upload form.
- `/predict`: Handles POST requests with image files, processes them, and returns the prediction.

### 3. Web Interface

**HTML Template (templates/index.html)**:
- Provides the layout for the user interface.
- Contains an image upload form and a section for displaying results.

**CSS (static/style.css)**:
- Styles the web page to ensure a clean and user-friendly interface.

**JavaScript (static/script.js)**:
- Handles the image upload process.
- Sends the image to the server for prediction using the Fetch API.
- Updates the web page with the uploaded image and prediction result.

## How It Works

### 1. Model Training

1. **Prepare Data**:
   - Organize images into class-specific directories.
   - Split data into training and validation sets.

2. **Define Model Architecture**:
   - Build a CNN with Conv2D, MaxPooling2D, Dropout, Flatten, and Dense layers.

3. **Compile and Train**:
   - Use the Adam optimizer and categorical crossentropy loss function.
   - Train the model on the training dataset while validating on the validation dataset.

4. **Save the Model**:
   - Export the trained model to a file for use in the Flask application.

### 2. Web Application

1. **Run Flask Server**:
   - Start the Flask application to host the web interface.

2. **Upload Image**:
   - Users select an image file and click "Predict."

3. **Process Image**:
   - Flask server preprocesses the image and feeds it into the trained model.

4. **Display Results**:
   - Show the uploaded image and the predicted class label on the web page.


