import streamlit as st
from fastai.vision.all import load_learner
from PIL import Image

st.title("🐻 Bear Classifier")
st.write("Upload an image of a Grizzly, Black, or Teddy bear to classify it.")


# Model loading with caching
@st.cache_resource
def load_model():
    return load_learner("bear_classifier.pkl")


model = load_model()

# Image uploader UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    pred, pred_idx, probs = model.predict(image)

    st.success(f"**Prediction:** {pred}")
    st.write(f"**Confidence:** {probs[pred_idx] * 100:.2f}%")