from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from mangum import Mangum 

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Route to serve the index.html file
@app.get("/")
async def serve_index():
    """Serve the index.html file when visiting the root endpoint."""
    if os.path.exists("websit/index.html"):
        return FileResponse("websit/index.html")
    return {"error": "❌ index.html not found!"}

# Global dictionaries to store loaded models and class labels
models = {}
class_labels = {}

def load_model(model_name):
    """Load a model only when needed."""
    if model_name in models:
        return models[model_name]  # Return the model if already loaded

    # Define model file paths
    model_files = {
        "best_model": "model/CNN.keras",
        "efficientnet": "model/InceptionV3_model.h5"
    }

    # Load the model if the file exists
    if model_name in model_files and os.path.exists(model_files[model_name]):
        models[model_name] = tf.keras.models.load_model(model_files[model_name])
        print(f"✅ Model {model_name} loaded successfully!")
        return models[model_name]
    else:
        print(f"❌ Model file for {model_name} not found!")
        return None

def load_class_labels(model_name):
    """Load class labels only when needed."""
    if model_name in class_labels:
        return class_labels[model_name]  # Return labels if already loaded

    # Define label file paths
    label_files = {
        "best_model": "websit/class_labels_best_model.json",
        "efficientnet": "websit/class_labels_inceptionv3.json"
    }

    # Load labels if the file exists
    if model_name in label_files and os.path.exists(label_files[model_name]):
        with open(label_files[model_name], "r") as f:
            class_labels[model_name] = json.load(f)
        class_labels[model_name] = {str(k): v for k, v in class_labels[model_name].items()}
        print(f"✅ Labels for {model_name} loaded successfully!")
        return class_labels[model_name]
    else:
        print(f"❌ Label file for {model_name} not found!")
        return {}

def preprocess_image(image: Image.Image):
    """Preprocess the image to make it suitable for the model."""
    image = image.convert("RGB")  # Convert to RGB format
    image = image.resize((224, 224))  # Resize to 224x224
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Route to handle image predictions
@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Query("best_model", enum=["best_model", "efficientnet"])
):
    """Receive an image, select the model, and return predictions."""
    try:
        print(f"📸 Received file: {file.filename}")

        # Read and preprocess the image
        image = Image.open(io.BytesIO(await file.read()))
        image = preprocess_image(image)

        # Load the model (if not already loaded)
        model = load_model(model_name)
        if model is None:
            return {"error": f"❌ Model {model_name} is not available!"}

        # Make predictions
        predictions = model.predict(image)
        predicted_class = str(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        # Load class labels (if not already loaded)
        model_classes = load_class_labels(model_name)
        predicted_label = model_classes.get(predicted_class, f"Unknown Class {predicted_class}")

        return {"model": model_name, "class": predicted_label, "confidence": confidence}
    
    except Exception as e:
        return {"error": str(e)}

# To make FastAPI work with Vercel and Render
handler = Mangum(app)