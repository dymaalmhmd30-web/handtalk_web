# HandTalk

HandTalk is a bilingual sign language translation system designed to support communication using both Arabic and English sign language alphabets. The system provides real-time sign-to-text and text-to-sign translation through an interactive web-based platform.

## Features

- Arabic Sign Language Recognition
- English Sign Language Recognition
- Real-Time Camera Prediction
- Image Upload Prediction
- Text-to-Sign Translation
- Web-Based Interactive Interface
- MediaPipe Hand Landmark Detection
- Deep Learning-Based Gesture Recognition

## Technologies Used

- Python
- Flask
- TensorFlow
- MediaPipe
- HTML
- CSS
- JavaScript
- Hugging Face Spaces

## Deep Learning Models

The project includes multiple deep learning and computer vision approaches for sign language recognition, including:

- CNN (Convolutional Neural Network)
- MobileNetV2
- MediaPipe + MLP Classifier

The final English recognition model is based on MediaPipe hand landmark extraction combined with an MLP classifier, while MobileNetV2 was utilized for Arabic sign language recognition.

## Project Structure

```text
handtalk-web/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── Procfile
├── Dockerfile
├── label_encoder.pkl
│
├── arabic_saved_model/
├── english_saved_model/
│
├── static/
└── templates/

## Dataset Sources

### Arabic Sign Language Datasets
- Arabic Alphabets Sign Language Dataset
- RGB Arabic Sign Language Dataset

### English Sign Language Dataset
- ASL Alphabet Dataset
Installation

Clone the repository:

git clone https://github.com/Deema190/handtalk-web.git

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py

Deployment

The system was deployed using Hugging Face Spaces with Docker support.

Live Demo

Hugging Face Deployment:

(https://huggingface.co/spaces/Deema190/handtalk-web)

Authors
