from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model = None

def get_model():
    global model
    if model is None:
        try:
            print("Loading MobileNetV2 from ImageNet...")
            model = tf.keras.applications.MobileNetV2(weights='imagenet')
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return model

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_model = get_model()
    if not img_model:
        return {"error": "Model is still downloading/loading for the first time. Please wait a few seconds and try again!"}
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        image = image.resize((224, 224))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
               
        prediction = img_model.predict(img_array)
        class_index = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][class_index])

        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=1)[0][0]
        real_object_name = decoded[1].replace('_', ' ').title()
        
        if 151 <= class_index <= 268:
            class_name = "Dog"
            breed = real_object_name
        elif 281 <= class_index <= 285:
            class_name = "Cat"
            breed = real_object_name
        else: