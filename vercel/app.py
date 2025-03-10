import streamlit as st
import requests
from PIL import Image
import io

# ✅ Set the app title
st.title("🔍 What is this? - Image Classification App 🐾")

# ✅ Dropdown to select the model
model_choice = st.selectbox(
    "📌 Choose the model you want to use:",
    ["best_model", "efficientnet"]
)

# ✅ Upload an image from the user
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ✅ Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # ✅ Send the image to FastAPI for prediction with the selected model
    with st.spinner("🤖 Analyzing the image..."):
        response = requests.post(
            f"http://127.0.0.1:8000/predict/?model_name={model_choice}",
            files={"file": uploaded_file.getvalue()}
        )

    # ✅ Display the prediction result
    if response.status_code == 200:
        prediction = response.json()
        st.success(f"🔍 Model: {prediction['model']}")
        st.success(f"🔍 This is: **{prediction['class']}**")
        st.info(f"🔹 Confidence Score: {prediction['confidence']:.2f}")
    else:
        st.error("❌ An error occurred during prediction")
