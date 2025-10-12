# emotion_detector.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
import os

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Model definition
# ----------------------------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128*7*7, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Load model
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(os.path.dirname(__file__), "best_emotion_cnn.pth")

model = EmotionCNN(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

EMOTIONS = ["angry", "happy", "sad", "stressed"]

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    img = Image.open(file.stream).convert("L")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
        predicted_emotion = EMOTIONS[pred.item()]

    return render_template("result.html", emotion=predicted_emotion)

@app.route("/home")
def go_home():
    return redirect(url_for("home"))

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
