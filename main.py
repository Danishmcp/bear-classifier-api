from fastapi import FastAPI, File, UploadFile
from fastai.vision.all import *
from PIL import Image
import io

app = FastAPI(title="Bear Species Classifier")

# Model load karo
learn = load_learner('bear_classifier.pkl')
categories = learn.dls.vocab

@app.get("/")
def home():
    return {"message": "Bear Classifier API is Live! Go to /docs to test images."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    pred, idx, probs = learn.predict(img)
    return {
        "prediction": str(pred),
        "confidence": float(probs[idx]),
        "probabilities": {cat: float(prob) for cat, prob in zip(categories, probs)}
    }