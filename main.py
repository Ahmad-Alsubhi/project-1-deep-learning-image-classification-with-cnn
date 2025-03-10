


from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from mangum import Mangum  # لجعل FastAPI يعمل مع Vercel

# ✅ إنشاء التطبيق
app = FastAPI()

# ✅ إضافة CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ تقديم index.html عند زيارة الصفحة الرئيسية
@app.get("/")
async def serve_index():
    """ تقديم ملف index.html عند زيارة الصفحة الرئيسية """
    if os.path.exists("websit/index.html"):
        return FileResponse("websit/index.html")
    return {"error": "❌ index.html not found!"}

# ✅ Lazy Loading للنماذج
models = {}
class_labels = {}

def load_model(model_name):
    """ تحميل النموذج فقط عند الحاجة """
    if model_name in models:
        return models[model_name]  # إذا كان محملاً مسبقًا، لا تعيد تحميله

    model_files = {
        "best_model": "model/CNN.keras",
        "efficientnet": "model/InceptionV3_model.h5"
    }

    if model_name in model_files and os.path.exists(model_files[model_name]):
        models[model_name] = tf.keras.models.load_model(model_files[model_name])
        print(f"✅ Model {model_name} loaded successfully!")
        return models[model_name]
    else:
        print(f"❌ Model file for {model_name} not found!")
        return None

def load_class_labels(model_name):
    """ تحميل تسميات الفئات فقط عند الحاجة """
    if model_name in class_labels:
        return class_labels[model_name]

    label_files = {
        "best_model": "websit/class_labels_best_model.json",
        "efficientnet": "websit/class_labels_inceptionv3.json"
    }

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
    """ تحويل الصورة لتنسيق مناسب للنموذج """
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
    """ استقبال الصورة، اختيار النموذج، وإرجاع التوقعات """
    try:
        print(f"📸 Received file: {file.filename}")

        # ✅ قراءة الصورة
        image = Image.open(io.BytesIO(await file.read()))
        image = preprocess_image(image)

        # ✅ تحميل النموذج (إذا لم يكن محملاً مسبقًا)
        model = load_model(model_name)
        if model is None:
            return {"error": f"❌ Model {model_name} is not available!"}

        # ✅ تنفيذ التوقعات
        predictions = model.predict(image)
        predicted_class = str(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        # ✅ تحميل التسميات (إذا لم تكن محملة مسبقًا)
        model_classes = load_class_labels(model_name)
        predicted_label = model_classes.get(predicted_class, f"Unknown Class {predicted_class}")

        return {"model": model_name, "class": predicted_label, "confidence": confidence}
    
    except Exception as e:
        return {"error": str(e)}

# ✅ لجعل FastAPI يعمل على Vercel و Render
handler = Mangum(app)
