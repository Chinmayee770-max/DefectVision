# Image Prediction App Using Custom CNN and VGG16 Models

## Overview

This project implements a web application that allows users to upload images for classification. The app uses two convolutional neural network (CNN) models—Custom CNN and VGG16—to predict whether a casting product is **defective** or **non-defective**. The results include the prediction, confidence scores from both models, and a comparison graph.

## Features
- **Dual Model Prediction**: Classifies images using both a custom-trained CNN and a pre-trained VGG16 model.
- **Confidence Comparison**: Displays the confidence percentages from both models.
- **Image Preview**: Displays the uploaded image along with the prediction results.
- **Graphical Representation**: A bar chart comparing the confidence scores of both models.
- **User-Friendly Interface**: Simple and intuitive web interface built using Flask.

## Prerequisites
Before running the application, make sure you have the following installed:

- Python 3.x
- TensorFlow
- Flask
- OpenCV
- NumPy
- Matplotlib
- Werkzeug

You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
Flask==2.0.3
tensorflow==2.11.0
opencv-python==4.5.5.64
numpy==1.23.3
matplotlib==3.6.2
Werkzeug==2.1.2
ImagePredictionApp/
│
├── app.py                 # Main application code
├── models/                # Folder containing the trained models
│   ├── casting_product_detection_normal.hdf5  # Custom CNN model
│   └── casting_product_detection_vgg16.hdf5   # VGG16 model
├── static/                # Folder for static assets (CSS, images, etc.)
│   ├── css/
│   │   └── styles.css     # Stylesheet for the web interface
│   └── uploads/           # Folder for uploaded images
├── templates/             # HTML templates
│   ├── upload.html        # Upload form
│   └── result.html        # Results page
└── requirements.txt       # List of dependencies
