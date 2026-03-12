import os
import base64
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import pytesseract
from transformers import pipeline


# ----------------------------
# Tesseract path
# ----------------------------
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract OCR\tesseract.exe"


# ----------------------------
# Flask
# ----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ----------------------------
# Sentiment model
# ----------------------------
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"
)


# ----------------------------
# CNN Model (same as training)
# ----------------------------
class EmotionCNN(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Flatten(),

            nn.Dropout(0.4),

            nn.Linear(128*7*7,256),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256,num_classes)

        )

    def forward(self,x):
        return self.net(x)


# ----------------------------
# Load model
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = os.path.join(os.path.dirname(__file__), "best_emotion_cnn.pth")

model = EmotionCNN().to(DEVICE)
model.load_state_dict(torch.load(model_path,map_location=DEVICE))
model.eval()


# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

EMOTIONS = ["angry","happy","sad","stressed"]


# ----------------------------
# OCR
# ----------------------------
def extract_text(img):

    try:
        img = img.convert("L")
        img = img.point(lambda x:0 if x<140 else 255,'1')
        text = pytesseract.image_to_string(img)
        return text.strip()
    except:
        return ""


# ----------------------------
# Segment Emotion Classification
# ----------------------------
def classify_text_segments(text):

    sentences = [s.strip() for s in text.split('.') if s.strip()]

    results = []

    for s in sentences:

        sentiment = sentiment_analyzer(s)[0]

        label = sentiment["label"].lower()

        if label == "positive":
            emotion = "happy"

        elif label == "negative":

            lower = s.lower()

            if "angry" in lower or "hate" in lower:
                emotion = "angry"

            elif "stress" in lower or "tired" in lower or "overwhelmed" in lower:
                emotion = "stressed"

            else:
                emotion = "sad"

        else:
            emotion = "sad"

        results.append({
            "text": s,
            "emotion": emotion
        })

    return results


# ----------------------------
# Mental health risk
# ----------------------------
def mental_health_risk(segments):

    negative_count = sum(1 for s in segments if s["emotion"] in ["sad","angry","stressed"])

    if negative_count >= 3:
        return "High"

    if negative_count == 2:
        return "Medium"

    return "Low"


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("home.html")


# ----------------------------
# Upload prediction
# ----------------------------
@app.route("/predict",methods=["POST"])
def predict():

    file = request.files["file"]

    filename = secure_filename(file.filename)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)

    file.save(filepath)

    img = Image.open(filepath).convert("L")

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        _,pred = torch.max(output,1)

    handwriting_emotion = EMOTIONS[pred.item()]

    text = extract_text(img)

    segments = classify_text_segments(text)

    risk = mental_health_risk(segments)

    return jsonify({
        "emotion": handwriting_emotion,
        "text": text,
        "segments": segments,
        "risk": risk,
        "image_url": f"/static/uploads/{filename}"
    })


# ----------------------------
# Canvas prediction
# ----------------------------
@app.route("/predict_canvas",methods=["POST"])
def predict_canvas():

    data = request.json["image"]

    image_data = data.split(",")[1]

    image_bytes = base64.b64decode(image_data)

    img = Image.open(BytesIO(image_bytes)).convert("L")

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        _,pred = torch.max(output,1)

    handwriting_emotion = EMOTIONS[pred.item()]

    text = extract_text(img)

    segments = classify_text_segments(text)

    risk = mental_health_risk(segments)

    return jsonify({
        "emotion": handwriting_emotion,
        "text": text,
        "segments": segments,
        "risk": risk,
        "image_url": ""
    })


# ----------------------------
# Static uploads
# ----------------------------
@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
