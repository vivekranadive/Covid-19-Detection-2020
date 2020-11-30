import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import pandas as pd
import cv2

model = load_model('covid19.model')
st.write("""
         Covid19 Detection through Lung Xrays
         """
         )
st.write("Detects if covid19 is present within the respitory system")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


def import_and_predict(image_data, model):

    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(224, 224),
                             interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img_resize[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if np.argmax(prediction) == 0:
        st.write("No Covid19 detected")
    elif np.argmax(prediction) == 1:
        st.write("Covid19 detected")

    st.text("Probability (0: healthy, 1: corona)")
    st.write(prediction)
