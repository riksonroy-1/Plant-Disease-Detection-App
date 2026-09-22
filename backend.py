from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI(title="Plant Disease Detection API")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = "model/plant_disease_model.keras"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None
    print(f"Warning: Model not found at {MODEL_PATH}")

# Load Class Names
# The original code used os.listdir on the training directory.
# Let's ensure the class names are loaded from the text file or directory
try:
    with open("model/class_names.txt", "r") as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    try:
        CLASS_NAMES = sorted(os.listdir("dataset/PlantVillage/train"))
    except FileNotFoundError:
        CLASS_NAMES = []
        print("Warning: Class names could not be loaded.")

@app.get("/")
def read_root():
    return {"message": "Plant Disease Detection API is running!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    if not CLASS_NAMES:
        raise HTTPException(status_code=500, detail="Class names are not loaded.")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess image
        img = image.resize((224, 224))
        img = np.array(img, dtype=np.float32)
        
        # Apply model specific preprocessing
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        
        # Predict
        prediction = model.predict(img, verbose=0)[0]
        
        predicted_index = int(np.argmax(prediction))
        confidence = float(prediction[predicted_index] * 100)
        
        predicted_class = CLASS_NAMES[predicted_index]
        
        if "___" in predicted_class:
            plant, disease = predicted_class.split("___")
        else:
            plant, disease = predicted_class, "Unknown"
            
        # Top 5 predictions
        top5_indices = np.argsort(prediction)[-5:][::-1]
        top5_predictions = [
            {"class": CLASS_NAMES[i].replace("___", " - "), "confidence": float(prediction[i] * 100)}
            for i in top5_indices
        ]
        
        return {
            "plant": plant,
            "disease": disease.replace("_", " "),
            "confidence": confidence,
            "top5": top5_predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
