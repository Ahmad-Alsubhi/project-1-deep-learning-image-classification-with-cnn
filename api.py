from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
import uvicorn

# ✅ Load Models
models = {
    "best_model": tf.keras.models.load_model("best_model_6.keras"),
    "efficientnet": tf.keras.models.load_model("InceptionV3_model.h5")
}
print("✅ All models loaded successfully!")

# ✅ Load Class Labels for Each Model
class_labels_files = {
    "best_model": "class_labels_best_model.json",   # 🔹 Labels for `best_model_6.keras`
    "efficientnet": "class_labels_inceptionv3.json"  # 🔹 Labels for `InceptionV3_model.h5`
}

class_labels = {}

# ✅ Load class labels separately for each model
for model_name, file_path in class_labels_files.items():
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            class_labels[model_name] = json.load(f)
        class_labels[model_name] = {str(k): v for k, v in class_labels[model_name].items()}
        print(f"✅ {model_name} class labels loaded successfully!")
    else:
        print(f"❌ Warning: {file_path} not found! Check your class label files.")

# ✅ Initialize FastAPI App
app = FastAPI()

# ✅ Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image: Image.Image):
    """ Preprocess image to match model input """
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Query("best_model", enum=["best_model", "efficientnet"])
):
    """ Receive an image, choose model, return prediction """
    try:
        print(f"Received file: {file.filename}")
        image = Image.open(io.BytesIO(await file.read()))
        image = preprocess_image(image)

        # ✅ Choose Model
        model = models.get(model_name)
        if model is None:
            return {"error": f"❌ Model {model_name} is not available!"}

        # ✅ Predict
        predictions = model.predict(image)
        predicted_class = str(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        # ✅ Choose Correct Class Labels
        model_classes = class_labels.get(model_name, {})
        predicted_label = model_classes.get(predicted_class, f"Unknown Class {predicted_class}")

        return {"model": model_name, "class": predicted_label, "confidence": confidence}
    
    except Exception as e:
        return {"error": str(e)}

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
