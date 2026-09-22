# Plant Disease Detection 🌿

A deep learning Plant Disease Detection system using MobileNetV2 and the PlantVillage dataset. It classifies leaf images into various diseases and includes a Streamlit web interface for interactive use, alongside a FastAPI backend for RESTful predictions.

## Features ✨

* **Streamlit Web Application (`app.py`)**: An easy-to-use web interface for users to upload plant leaf images and view predictions, including the specific plant, the disease, confidence scores, and the top 5 closest predictions.
* **FastAPI Backend (`backend.py`)**: A high-performance RESTful API that provides a `/predict` endpoint. It accepts image uploads and returns classification results in JSON format, perfect for integrating into mobile apps or other services.
* **Deep Learning Model (`train.py`)**: Utilizes the MobileNetV2 model (pre-trained on ImageNet) for efficient, accurate, and lightweight image classification.
* **Robust Training Pipeline**: Includes data augmentation (random flips, rotations, and zooms) to improve the model's ability to generalize to new, unseen images.

## Tech Stack 🛠️

* **Python**
* **TensorFlow & Keras**: For building, training, and running the deep learning model.
* **Streamlit**: For the interactive frontend dashboard.
* **FastAPI & Uvicorn**: For the backend REST API server.
* **Pillow (PIL) & NumPy**: For image loading and array manipulations.

## Project Structure 📁

```text
PlantDiseaseDetection/
│
├── app.py                 # Streamlit frontend application
├── backend.py             # FastAPI backend API
├── train.py               # Script for training the deep learning model
├── requirements.txt       # Python dependencies
├── dataset/               # Directory for the PlantVillage dataset
│   └── PlantVillage/
│       ├── train/         # Training images
│       └── val/           # Validation images
├── model/                 # Directory containing the saved model
│   ├── plant_disease_model.keras
│   └── class_names.txt
├── notebooks/             # Directory for exploratory Jupyter notebooks
└── images/                # Example/test images (if any)
```

## Installation & Setup 🚀

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd PlantDiseaseDetection
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage 💡

### 1. Running the Streamlit App
To launch the interactive web interface, run:
```bash
streamlit run app.py
```
This will open the application in your default web browser (usually at `http://localhost:8501`).

### 2. Running the FastAPI Backend
To start the REST API server, run:
```bash
uvicorn backend:app --reload
# OR simply:
python backend.py
```
The API will be available at `http://localhost:8000`. You can access the interactive Swagger documentation at `http://localhost:8000/docs`.

### 3. Training the Model (Optional)
If you want to train the model from scratch, ensure your dataset is placed in `dataset/PlantVillage/train` and `dataset/PlantVillage/val`, then run:
```bash
python train.py
```
This will train the MobileNetV2 model and save the best weights to the `model/` directory along with `class_names.txt`.

## Dataset 📊

This project relies on the **PlantVillage** dataset. If you are training the model yourself, make sure your directory structure matches what `train.py` expects:

```
dataset/
└── PlantVillage/
    ├── train/
    │   ├── Apple___Apple_scab/
    │   ├── Apple___healthy/
    │   └── ...
    └── val/
        ├── Apple___Apple_scab/
        ├── Apple___healthy/
        └── ...
```
