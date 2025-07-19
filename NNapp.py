# Step 10: Streamlit prediction tool (save as app.py)
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load the saved model
model = load_model('mnist_model.h5')

st.title("MNIST Digit Classifier")
st.write("Upload a 28×28 grayscale image of a handwritten digit.")

uploaded_file = st.file_uploader("Choose an image", type=["png","jpg"])
if uploaded_file:
    # Open, resize, invert (if needed), and normalize
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28,28))
    img = ImageOps.invert(img)  # MNIST digits are white on black
    st.image(img, caption="Input (28×28)", width=100)
    
    # Prepare for prediction
    x = np.array(img).reshape(1, 784).astype('float32') / 255.0
    preds = model.predict(x)
    st.write(f"Predicted digit: **{np.argmax(preds)}**")

