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
