# Mold_Prediction_App
# Mold Detection and Prevention App
Software -Powered Mold Prediction Using Image Classification

## Project Overview
This is a web-based machine learning application designed to detect mold presence from uploaded images. Users can upload indoor wall images and receive predictions indicating whether mold is present.

## Key Features
- User authentication system (Register and Login)
- Image upload functionality with secure file handling
- User fills in the loaction, ventilation type, water leakage history or any respiratory health issue 
- Deep learning model for image classification (Mold vs Clean)
- Confidence score included with prediction output
- Database storage using SQLite

## Machine Learning Model Details
- Convolutional Neural Network (CNN) trained for mold detection
- Implemented using TensorFlow/Keras
- Model is stored in the 'model/model.h5' file

## Environment Setup
Clone the repository, install necessary dependencies, and run the application.

commands:
git clone <https://github.com/WanPgui/Mold_Prediction_App.git>
cd mold-prediction-app
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
flask run

Visit the following local URL to use the app:
http://127.0.0.1:5000/

## Deployment Plan
1. Local deployment with Flask development server
2. cloud deployment on Render
3. Optional improvements including CI/CD pipeline and HTTPS security

## Project Structure
mold-prediction-app/
|-- README.md
|-- app.py
|-- requirements.txt
|-- mold_model_final.keras (placeholder)
|-- templates (HTML UI files)
|-- static/css and static/images (UI assets)
|-- uploads (uploaded user images)
|-- designs/screenshots and mockups (design documentation)

## Video Demonstration Guidelines
Link: https://drive.google.com/file/d/1LjBlqI5sX2ReWBvGCZBoyNqiQzJ5UKOX/view?usp=sharing


## Code file
https://colab.research.google.com/drive/1xinoPeZNUUXozQKzYDKinpG9DSfA1BGU?usp=sharing


## Developer Information
Name: Peris Wangui
Course and Institution: Bse, ALU
