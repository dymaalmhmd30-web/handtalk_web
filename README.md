# HandTalk

HandTalk is a bilingual sign language translation system designed to support communication using both Arabic and English sign language alphabets. The system provides real-time sign-to-text and text-to-sign translation through an interactive web-based platform.

## Features

- Arabic Sign Language Recognition
- English Sign Language Recognition
- Real-Time Camera Prediction
- Image Upload Prediction
- Text-to-Sign Translation
- MediaPipe Hand Landmark Detection
- Web-Based Interactive Interface

## Technologies Used

- Python
- Flask
- TensorFlow
- MediaPipe
- HTML
- CSS
- JavaScript
- Hugging Face Spaces

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
```

## Dataset Sources

### Arabic Sign Language Datasets
- Arabic Alphabets Sign Language Dataset
- RGB Arabic Sign Language Dataset

### English Sign Language Dataset
- ASL Alphabet Dataset

Datasets were obtained from publicly available Kaggle sources.


## Installation

Clone the repository:

```bash
git clone https://github.com/Deema190/handtalk-web.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

## Deployment

The system was deployed using Hugging Face Spaces with Docker support.

## Live Demo

Hugging Face Deployment:

https://huggingface.co/spaces/Deema190/handtalk-web

## Dataset Download:

Dataset 1
📁 [Google Drive Folder](https://drive.google.com/drive/folders/1uCfSaOPuiUjue3zUnzA_DyTMgcuTtstb?usp=sharing)

Dataset 2
📁 [Google Drive Folder](https://drive.google.com/drive/folders/1gBS3SGrezm0mkxRensDPfkucsK7ICNZG?usp=sharing)

## Authors

- Deema Almohamad
- Rimas Almalki 
- Rawyah Alasmari
- Sara Alzolafy
