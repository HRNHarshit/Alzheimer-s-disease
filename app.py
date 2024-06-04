import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Class names
class_names = ['Mild Dementia', 'Moderate Dementia', 'Non Dementia', 'Very Mild Dementia']

# Disable file uploader encoding warning
st.set_option('deprecation.showfileUploaderEncoding', False)

# Load model function with new caching decorator
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('./alzheimer_model.keras')
    return model

# Load the model
model = load_model()

# Custom CSS to add a white background
st.markdown(
    """
    <style>
    .main {
        background-color: white;
        padding-left: 5%;
        padding-right: 5%;
        border: 1px solid #ccc;
    }
    h1, h2, h3, h4, h5, h6, .block-container {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app title
st.title("Alzheimer's Disease Detection")
st.write("Please upload an image file for Alzheimer's disease prediction.")

# File uploader
file = st.file_uploader("", type=["jpg", "png"])

def import_and_predict(image_data, model):
    image = ImageOps.exif_transpose(image_data)
    img = image.convert('RGB')
    img_reshape = np.array(img)[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("No image uploaded.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.subheader("Prediction Results:")
    st.write(dict(zip(class_names, predictions[0])))
    st.subheader("Prediction Summary:")
    predicted_class = class_names[np.argmax(predictions[0])]
    accuracy = round(100 * predictions[0][np.argmax(predictions[0])], 2)
    toPrint = "Predicted class: {} with {}% accuracy, ".format(predicted_class, accuracy)
    
    if predicted_class == 'Mild Dementia':
        toPrint += "which means there is a mild chance of Alzheimer's disease."
    elif predicted_class == 'Moderate Dementia':
        toPrint += "which means there is a moderate chance of Alzheimer's disease."
    elif predicted_class == 'Non Dementia':
        toPrint += "which means there is a no chance of Alzheimer's disease."
    elif predicted_class == 'Very Mild Dementia':
        toPrint += "which means there is a very mild chance of Alzheimer's disease."
    st.write(toPrint)
