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

## Requirements

You can install the required dependencies by running the following command:



pip install -r requirements.txt
Flask==2.0.3
tensorflow==2.11.0
opencv-python==4.5.5.64
numpy==1.23.3
matplotlib==3.6.2
Werkzeug==2.1.2


## File Structure

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

## Setup and Running the Application
### Clone the repository:

bash
### Copy code
git clone https://github.com/your-username/ImagePredictionApp.git
cd ImagePredictionApp
### Install the dependencies:

bash
### Copy code
pip install -r requirements.txt
Place your models in the models/ folder:

### Make sure that your casting_product_detection_normal.hdf5 (Custom CNN) and casting_product_detection_vgg16.hdf5 (VGG16) models are placed in the models/ folder.
### Run the application:

bash
### Copy code
python app.py
The application will start running on http://127.0.0.1:5000/.

### Upload and predict:

Open your web browser and go to http://127.0.0.1:5000/.
Use the upload form to select an image of a casting product.
The app will classify the image as either Defective or Non-defective and display the prediction results along with confidence scores for both models.
How It Works
Upload Image:

Users upload an image via a simple form on the main page (upload.html).
### Image Processing:

The uploaded image is processed (resized, normalized) before being fed into both the Custom CNN and VGG16 models for prediction.
### Prediction:

Both models return a prediction value between 0 and 1, which indicates the likelihood of the image being defective.
If the output is greater than 0.5, the product is classified as defective; otherwise, it is classified as non-defective.
### Confidence Display:

Confidence percentages for both models are calculated based on the prediction scores and are displayed to the user.
### Confidence Graph:

A bar chart is generated to visually compare the confidence values of both models.
### Result Page:

The results are displayed on the result.html page, which includes the uploaded image, the final prediction, confidence percentages, and the confidence comparison graph.
Customization
### Models: Replace casting_product_detection_normal.hdf5 and casting_product_detection_vgg16.hdf5 with your own models if needed.
### Image Preprocessing: Modify the prepare_image function to adjust the image preprocessing steps according to your model's requirements (e.g., changing image resizing dimensions or normalization methods).
### Styling: You can customize the styling by editing static/css/styles.css.
### Limitations and Future Work
Model Overfitting: The custom CNN model might overfit if trained on a small dataset. Expanding the training dataset or using techniques like data augmentation could improve generalization.
Real-Time Predictions: This app could be extended to handle real-time image streams for automatic predictions.
Model Improvement: Further refinement of both models is recommended to enhance the prediction accuracy, potentially using more advanced architectures like ResNet or EfficientNet.
